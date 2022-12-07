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
from tqdm.auto import tqdm
from random import choice


def gaussian_p_sample_loop(diffusion_model, token, mask, shape, num_timesteps, tokenizer, text_ctx,
                           noise=None, progress=False, dtype=mindspore.float32, vocab_len=50001):
    # init original image(pure noise)
    if noise is not None:
        img = noise
    else:
        img = mindspore.ops.StandardNormal()(shape)
        img = mindspore.ops.Cast()(img, dtype)
    indices = list(range(num_timesteps))[::-1]

    # visualized progress bar
    if progress:
        indices = tqdm(indices)

    # recursively de-noising on img
    for i in indices:
        random_token_tensor = mindspore.numpy.randint(1, vocab_len-1, (text_ctx,), dtype=mindspore.int32)
        random_mask_tensor = mindspore.numpy.ones((text_ctx,), mindspore.int32)
        i_tensor = mindspore.Tensor([i], dtype=mindspore.int32)
        sample, _ = diffusion_model(x=img, timesteps=i_tensor, token=token, mask=mask,
                                    random_token=random_token_tensor, random_mask=random_mask_tensor)
        img = sample

    return img


def ddim_sample_loop(super_res_model, up_shape, samples, token, mask, num_timesteps, noise=None, progress=False,
                     dtype=mindspore.float32):
    # init original image(pure noise)
    if noise is not None:
        img = noise
    else:
        upsample_temp = 0.997
        img = mindspore.ops.StandardNormal()(up_shape)
        img = mindspore.ops.Mul()(img, upsample_temp)
        img = mindspore.ops.Cast()(img, dtype)

    indices = list(range(num_timesteps))[::-1]

    # visualized progress bar
    if progress:
        indices = tqdm(indices)

    for i in indices:
        i_tensor = mindspore.Tensor(input_data=[i], dtype=mindspore.int32)
        sample, _ = super_res_model(x=img, timesteps=i_tensor, token=token, mask=mask, samples=samples)
        img = sample

    return img

