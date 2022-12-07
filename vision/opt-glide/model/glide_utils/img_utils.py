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

import mindspore
from typing import Tuple
import numpy as np
from PIL import Image
from IPython.display import display


def read_image(path: str, size: int = 256) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    # print("img.shape", img.shape) [64, 64, 3]
    dimmed = mindspore.Tensor(img)[None]
    reshaped = dimmed.transpose((0, 3, 1, 2))
    reshaped = mindspore.ops.Cast()(reshaped, mindspore.float32)
    scaled = mindspore.ops.Add()(mindspore.ops.Div()(reshaped, 127.5), -1)
    return scaled


def get_img(batch: mindspore.Tensor):
    batch_plus = mindspore.ops.Add()(batch, 1)
    scaled = mindspore.ops.Mul()(batch_plus, 127.5)
    rounded_scaled = mindspore.ops.Rint()(scaled)
    clipped_scaled = mindspore.ops.clip_by_value(rounded_scaled, mindspore.Tensor(0), mindspore.Tensor(255))
    clipped_scaled = clipped_scaled.transpose((2, 0, 3, 1))
    clipped_scaled = mindspore.ops.Cast()(clipped_scaled, mindspore.uint8)
    reshaped = clipped_scaled.reshape(([batch.shape[2], -1, 3]))
    return reshaped


def show_images(batch: mindspore.Tensor):
    """ Display a batch of images inline. """
    display(Image.fromarray(get_img(batch).asnumpy()))


def save_images(batch: mindspore.Tensor, path: str):
    """ Display a batch of images inline. """
    batch_32 = mindspore.ops.Cast()(batch, mindspore.float32)
    Image.fromarray(get_img(batch_32).asnumpy()).save(path, quality=100, subsampling=0)
