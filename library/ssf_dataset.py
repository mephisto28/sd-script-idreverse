import os
import json
import math
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
from sbp.utils.logging import logger
from sbp.vision.protocal import ImageResult
from sbp.io.common.encoder_decoder import decode_b64
from sbp.io import SequenceFileReader, SsfRandomReader, read_map
from sbp.utils.parallel import parallel_imap

from library.train_util import BaseDataset, BaseSubset, ImageInfo, DatasetGroup


@dataclass
class SsfSubsetConfig:

    image_ssf: str
    caption_map: str
    condition_ssfs: dict = field(default_factory=dict)
    meta_ssfs: dict = field(default_factory=dict)
    num_repeats: int = 1
    shuffle_caption: bool = False
    keep_tokens: int = 1
    color_aug: bool = False
    flip_aug: bool = False
    face_crop_aug_range = None
    random_crop: bool = False
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    caption_tag_dropout_rate: float = 0.0
    token_warmup_min: int = 0
    token_warmup_step: int = 0
    class_tokens: str = None
    is_reg: bool = False

    def __post_init__(self):
        self.image_dir = self.image_ssf
        self.img_count = 0


@dataclass
class SsfDatasetConfig:

    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        assert 'datasets' in self.config
        self.datasets = [SsfSubsetConfig(**dataset) for dataset in self.config['datasets']]
        self.enable_bucket = self.config['enable_bucket']
        self.resolution = self.config['resolution']

    def __getattr__(self, key):
        return self.config[key]


def get_sinusoidal_positional_encoding(position, dimension):
    encoding = [0.0] * dimension
    for i in range(dimension):
        if i % 2 == 0:
            encoding[i] = math.sin(position * (10000 ** (2*i / dimension)))
        else:
            encoding[i] = math.cos(position * (10000 ** ((2*i - 1) / dimension)))
    return encoding


def encode_pos_scale(bbox, h, w):
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / w / 2
    y = (y1 + y2) / h / 2
    s = (y2 - y1 + x2 - x2) / (w + h) * 2
    x_enc = get_sinusoidal_positional_encoding(x, 64)
    y_enc = get_sinusoidal_positional_encoding(y, 64)
    s_enc = get_sinusoidal_positional_encoding(s, 64)
    return np.array([x_enc, y_enc, s_enc]).flatten()


def decode_feature_new(s):
    result = ImageResult.from_json(json.loads(s))
    face_results = result.get_detection('face', topk=3)
    for i, face in enumerate(face_results):
        face_feature = face.rec_feature
        return face_feature.reshape([1, 512])


def decode_feature_and_pos(s):
    result = ImageResult.from_json(json.loads(s))
    h, w = result.height, result.width
    face_results = result.get_detection('face', topk=3)
    results = np.zeros([3, 512+64*3], dtype=np.float32)
    for i, face in enumerate(face_results):
        pos_enc = encode_pos_scale(face.bbox, h, w)
        face_feature = face.rec_feature
        results[i] = np.concatenate([pos_enc, face_feature])
    return results


def load_meta_v0_id_and_hw(task):
    path, key, pos = task
    s = SsfRandomReader.read(path, pos)
    result = ImageResult.from_json(json.loads(s))
    face_results = result.get_detection('face', topk=3)
    for i, face in enumerate(face_results):
        face_feature = face.rec_feature[None, ...]
        return {
            'face':face_feature, 
            'height': result.height, 
            'width': result.width
        }
    return {
        'face': np.zeros([1, 512], dtype=np.float16),
        'height': result.height,
        'width': result.width
    }
          

def load_image_metas(ksp):
    for result in tqdm(parallel_imap(load_meta_v0_id_and_hw, ksp), total=len(ksp)):
        yield result


class DreamBoothSsfDataset(BaseDataset):
    def __init__(
        self,
        data_config,
        tokenizer,
        max_token_length,
        debug_dataset,
    ) -> None:
        self.config = SsfDatasetConfig(data_config)
        super().__init__(tokenizer, max_token_length, self.config.resolution, debug_dataset)
        self.batch_size = self.config.batch_size
        self.size = resolution = self.config.resolution  # 短いほう
        self.prior_loss_weight = 1
        self.latents_cache = None
        self.caption_dict = {}
        self.enable_bucket = self.config.enable_bucket
        if self.enable_bucket:
            assert (
                min(resolution) >= self.config.min_bucket_reso
            ), f"min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
            assert (
                max(resolution) <= self.config.max_bucket_reso
            ), f"max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"
            self.min_bucket_reso = self.config.min_bucket_reso
            self.max_bucket_reso = self.config.max_bucket_reso
            self.bucket_reso_steps = self.config.bucket_reso_steps
            self.bucket_no_upscale = self.config.bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False
        self.feature_db = []
        self.load_meta()

    def load_meta(self,):
        dataset_config = self.config
        for dataset_no, dataset in enumerate(dataset_config.datasets):
            image_ssf = SequenceFileReader(dataset.image_ssf)
            id_feature_ssf = SequenceFileReader(dataset.meta_ssfs['face'])
            captions = read_map(dataset.caption_map, )
            keys = set(image_ssf.keys)
            for k in (id_feature_ssf.keys, captions.keys()):
                keys = keys.intersection(k)
                logger.info("found %d/%d keys intersection in %s" % (len(keys), len(image_ssf.keys), dataset.image_ssf))
            keys = sorted(list(keys))
            id_keypos = id_feature_ssf.zip_key_with_pos(keys=keys)
            img_keypos = image_ssf.zip_key_with_pos(keys=keys)
            metas = [meta for meta in load_image_metas(id_keypos)]
            for i, k in enumerate(keys):
                meta = metas[i]
                info = ImageInfo(
                    image_key = (dataset_no, k),
                    caption = captions[k],
                    absolute_path = img_keypos[i],
                    extra_feature = meta['face'],
                    num_repeats = dataset.num_repeats,
                    is_reg = False,
                    image_size = (meta['width'], meta['height']) if meta['height'] is not None else None
                )
                self.feature_db.append(meta['face'][0].astype(np.float16))
                self.register_image(info, dataset)
            dataset.img_count = len(keys)
            self.subsets.append(dataset)

        num_train_images = sum([subset.img_count for subset in self.subsets])
        logger.info(f"{num_train_images} train images with repeating.")
        self.num_train_images = num_train_images
        self.num_reg_images = 0

    def load_image(self, image_path):
        path, k, pos = image_path
        img = SsfRandomReader.read_image(path, pos)[..., ::-1].copy()
        return img


if __name__ == '__main__':
    from dataclasses import asdict
    from transformers import AutoTokenizer     
    tokenizer = AutoTokenizer.from_pretrained(
        # "C:/Users/admin/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0",
        "/home/zwshi/.cache/huggingface/hub/models--emilianJR--chilloutmix_NiPrunedFp32Fix/snapshots/4688d3087e95035d798c2b65cc89eeefcb042906/",
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    seed = 0
    dataset = DreamBoothSsfDataset("experiments/e001/config.json", tokenizer, 221, False)
    dataset.make_buckets()
    dataset.set_seed(seed)
    data_group = DatasetGroup([dataset])