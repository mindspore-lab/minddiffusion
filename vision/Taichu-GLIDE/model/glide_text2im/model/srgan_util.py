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

from mindspore import ops
from PIL import Image
import numpy as np
import mindspore
from mindspore import load_checkpoint, load_param_into_net, Tensor

from .srgan import Generator


def get_img(batch: mindspore.Tensor):
    batch_plus = mindspore.ops.Add()(batch, 1)
    scaled = mindspore.ops.Mul()(batch_plus, 127.5)
    rounded_scaled = mindspore.ops.Rint()(scaled)
    clipped_scaled = mindspore.ops.clip_by_value(rounded_scaled, mindspore.Tensor(0), mindspore.Tensor(255))
    clipped_scaled = clipped_scaled.transpose((2, 0, 3, 1))
    clipped_scaled = mindspore.ops.Cast()(clipped_scaled, mindspore.uint8)
    reshaped = clipped_scaled.reshape(([batch.shape[2], -1, 3]))
    return reshaped


class SRGAN():
    def __init__(self, upscale_factor, ckpt_path):
        self.net = Generator(upscale_factor)
        params = load_checkpoint(ckpt_path)
        load_param_into_net(self.net, params)
        self.reduce_dims = ops.ReduceSum(keep_dims=False)
        self.expand_dims = ops.ExpandDims()

    # SR from Tensor
    def sr_handle(self, lr):
        output = self.net(lr)
        return output

    # SR from image
    def sr_image(self, lr_image, hr_image):
        lr = np.array(Image.open(lr_image).convert("RGB"))
        lr = (lr / 127.5) - 1.0
        lr = lr.transpose(2, 0, 1).astype(np.float32)
        lr = np.expand_dims(lr, axis=0)
        output = self.sr_handle(Tensor(lr))
        output = output.asnumpy()
        output = np.squeeze(output, axis=0)
        output = np.clip(output, -1.0, 1.0)
        output = (output + 1.0) / 2.0
        output = output.transpose(1, 2, 0)
        Image.fromarray((output * 255.0).astype(np.uint8)).save(hr_image, quality=100)
