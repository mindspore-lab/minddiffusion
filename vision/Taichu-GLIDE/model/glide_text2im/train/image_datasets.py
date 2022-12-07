# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import re
import math
import random
from random import randint, choice

from tqdm import tqdm
from PIL import Image
import blobfile as bf
import numpy as np
import mindspore as ms
import mindspore.dataset as md
from mindspore.dataset import GeneratorDataset, DistributedSampler
import mindspore.dataset.vision.c_transforms as CV
from mindspore.communication.management import get_rank, get_group_size

from model.glide_text2im.train.t2ids import t2i_collate,t2i_collate_supres
from model.glide_text2im.train.Loader import MetaLoader, SupresMetaLoader
import model.glide_text2im.train.logger as logger
from model.glide_text2im.train.resample import create_named_schedule_sampler
from model.glide_text2im.train.sampler import BatchSampler
from model.glide_text2im.train.data_loader import DataLoader
from model.glide_text2im.train.generator import data_column,data_column_supres


data_len = 0
data_len_per_gpu = 0
epoch = 0

def load_data(
    *,
    data_path,
    image_caption_path_file,
    timesteps,
    batch_size,
    image_size,
    tokenizer,
    text_ctx,
    text_drop_p,
    resolution_ori,
    using_data_sampler=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    sampler_name="uniform",
    is_super_res=False,
    device_num=1,
    device_id=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not image_caption_path_file:
        raise ValueError("unspecified data directory")
    all_files, all_captions = _list_image_files_captions_recursively(image_caption_path_file, data_path)

    logger.log("len(all_files)", len(all_files))
    logger.log("len(all_captions)", len(all_captions))
    logger.log("all_files[0]", all_files[0])
    logger.log("all_captions[0]", all_captions[0])

    global data_len
    data_len = len(all_files)
    global data_len_per_gpu
    data_len_per_gpu = data_len

    dataloaders = {}
    if is_super_res:
        dataset = ImageDatasetSupres(
        image_size,
        resolution_ori,
        all_files,
        all_captions,
        tokenizer,
        text_ctx,
        text_drop_p,
        timesteps,
        shard=0,
        num_shards=1,
        shuffle=True,
        random_crop=False,
        random_flip=False,
        sampler_name="uniform"
        )
        
        datalen = dataset.__len__
        loader = build_dataloader_ft(dataset, datalen, t2i_collate_supres, batch_size, device_num)
        dataloaders["ftT2I"] = loader
        batchlen = datalen//(batch_size * device_num)
        metaloader = SupresMetaLoader(dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()))
        dataset = GeneratorDataset(metaloader, column_names=data_column_supres, shuffle=True)
    else:
        dataset = ImageDataset(
            batch_size,
            image_size,
            all_files,
            all_captions,
            tokenizer,
            text_ctx,
            text_drop_p,
            timesteps,
            shard=0,
            num_shards=1,
            random_crop=random_crop,
            random_flip=random_flip,
            sampler_name="uniform",
        )
        datalen = dataset.__len__
        loader = build_dataloader_ft(dataset, datalen, t2i_collate, batch_size, device_num)
        dataloaders["ftT2I"] = loader
        batchlen = datalen//(batch_size * device_num)
        metaloader = MetaLoader(dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()))
        dataset = GeneratorDataset(metaloader, column_names=data_column, shuffle=True)

    return dataset


def getfiles(dirPath, type='.*\.txt'):
    fileList = []
    if dirPath is None:
        return fileList
    # open directory
    files = os.listdir(dirPath)

    ptn = re.compile(type)
    for f in files:
        if (os.path.isfile(os.path.join(dirPath, f))):
            res = ptn.match(f)
            if (res != None):
                fileList.append(os.path.join(dirPath, res.group()))

    return fileList


def _list_image_files_captions_recursively(data_dir, data_path):

    if os.path.isdir(data_dir):
        image_files_captions_files = getfiles(data_dir)
        print("don't support to read from file now")
    else:
        with open(data_dir, 'r', encoding='utf-8') as file:
            content_list = file.readlines()
            image_files_captions_files = []
            image_files_captions_files_roots = []
            for x in content_list:
                if not x.strip().startswith("#"):
                    datas = x.split(" ")
                    if len(datas) != 2:
                        print("format error:", x)
                        continue
                    file = datas[0]
                    file = os.path.join(data_path, file)
                    root = datas[1]
                    root = os.path.join(data_path, root)
                    image_files_captions_files.append(file.strip())
                    image_files_captions_files_roots.append(root.strip())

    print('getting image files captions......')
    image_files = []
    captions = []
    for image_files_captions_file, image_files_captions_files_root in zip(image_files_captions_files, image_files_captions_files_roots):
        with open(image_files_captions_file, encoding='utf-8') as image_files_captions_file_fp:
            image_files_captions_file_fp_lines = image_files_captions_file_fp.readlines()
            for line in tqdm(image_files_captions_file_fp_lines):
                s = line.strip().split('\t')
                # order is different from cc3m
                if len(s) == 2:
                    path = s[0]
                    caption = s[1]
                    image_files.append(os.path.join(image_files_captions_files_root, path))
                    captions.append(caption)

                elif len(s) > 2:
                    path = s[0]
                    caption = "\t".join(s[1:])
                    image_files.append(os.path.join(image_files_captions_files_root, path))
                    captions.append(caption)
    return image_files, captions

class ImageDataset():
    def __init__(
        self,
        batch_size,
        resolution,
        image_paths,
        captions,
        tokenizer,
        text_ctx,
        text_drop_p,
        timesteps,
        shard=0,
        num_shards=1,
        shuffle=True,
        random_crop=False,
        random_flip=False,
        sampler_name="uniform",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.text_ctx = text_ctx
        self.text_drop_p = text_drop_p
        self.local_images = image_paths[shard:][::num_shards]
        self.local_captions = captions[shard:][::num_shards]
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.image_transform = md.transforms.Compose([
            md.vision.Decode(),
            md.vision.RandomResizedCrop(resolution,
                                scale=(1., 1.),
                                ratio=(1., 1.)),
            md.vision.ToTensor()
        ])
        self.sampler = create_named_schedule_sampler(sampler_name,timesteps)
    @property
    def __len__(self):
        return len(self.local_images)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        f = bf.BlobFile(path, "rb")
        pil_image = Image.open(f)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        caption = self.local_captions[idx]
        captions = caption.split("\t")
        if len(captions) >= 2:
            caption = choice(captions)

        # tokenizer
        tokens = self.tokenizer.encode(caption)
        if random.random() < self.text_drop_p:
            tokens = []
        tokens, mask = self.tokenizer.padded_tokens_and_mask(
            tokens, self.text_ctx  # text_ctx 128
        )

        t, weight = self.sampler.sample(1)
        return np.transpose(arr, [2,0,1]), np.array(tokens,dtype=np.int32), np.array(mask, dtype=np.int32), t[0].astype(np.int32), weight[0].astype(np.float32)

class ImageDatasetSupres():
    def __init__(
        self,
        resolution,
        resolution_ori,
        image_paths,
        captions,
        tokenizer,
        text_ctx,
        text_drop_p,
        timesteps,
        shard=0,
        num_shards=1,
        shuffle=True,
        random_crop=False,
        random_flip=False,
        sampler_name="uniform"
    ):
        super().__init__()
        self.resolution = resolution
        self.resolution_ori = resolution_ori
        self.tokenizer = tokenizer
        self.text_ctx = text_ctx
        self.text_drop_p = text_drop_p
        self.local_images = image_paths[shard:][::num_shards]
        self.local_captions = captions[shard:][::num_shards]
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.image_transform = md.transforms.Compose([
            md.vision.Decode(),
            md.vision.RandomResizedCrop(resolution,
                                scale=(1., 1.),
                                ratio=(1., 1.)),
            md.vision.ToTensor()
        ])
        
        self.sampler = create_named_schedule_sampler(sampler_name,timesteps)
    
    @property
    def __len__(self):
        return len(self.local_images)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        try:
            f = bf.BlobFile(path, "rb")
            pil_image = Image.open(f)
            pil_image.load()
        except Exception as e:
            return self.skip_sample(idx)

        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
            arr_ori = center_crop_arr(pil_image, self.resolution_ori)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            arr_ori = arr_ori[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        arr_ori = arr_ori.astype(np.float32) / 127.5 - 1

        caption = self.local_captions[idx]
        captions = caption.split("\t")
        if len(captions) >= 2:
            caption = choice(captions)

        # tokenizer
        tokens = self.tokenizer.encode(caption)
        if random.random() < self.text_drop_p:
            tokens = []
        tokens, mask = self.tokenizer.padded_tokens_and_mask(
            tokens, self.text_ctx  # text_ctx 128
        )

        # Pack the tokens together into model kwargs.
        low_res=np.transpose(arr_ori, [2, 0, 1])

        tokens, mask = self.tokenizer.padded_tokens_and_mask(
            tokens, self.text_ctx  # text_ctx 128
        )

        t, weight = self.sampler.sample(1)
        return np.transpose(arr, [2, 0, 1]), np.array(tokens,dtype=np.int32), np.array(mask, dtype=np.int32), t[0].astype(np.int32), weight[0].astype(np.float32), low_res


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def build_dataloader_ft(dataset, datalens,collate_fn, batch_size, device_num):
    sampler = BatchSampler(datalens, batch_size=batch_size, device_num=device_num)
    loader = DataLoader(dataset, batch_sampler=sampler, is_train=True, collate_fn=collate_fn, device_num=device_num,drop_last=True)
    return loader

if __name__ == "__main__":
    print("run")
