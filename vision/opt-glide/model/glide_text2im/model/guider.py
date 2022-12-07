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

import mindspore.nn as nn
import mindspore


class CombinePrompt(nn.Cell):
    def __init__(self, pics_generated):
        super(CombinePrompt, self).__init__()
        # model attributes
        self.pics_generated = pics_generated

        # operations
        self.slice = mindspore.ops.Slice()
        self.concat = mindspore.ops.Concat(axis=0)
        self.broadcast_to = mindspore.ops.BroadcastTo((pics_generated, 128))
        self.cast = mindspore.ops.Cast()

    '''
    x_t: tensor
    kwargs: dict, {tokens : num_of_pics*2 x 128 tensor, mask : num_of_pics*2 x 128 tensor}
    '''
    def construct(self, x_t, in_token, in_mask, random_token, random_mask):
        # computes
        first_half_tokens = self.slice(in_token, (0, 0), (self.pics_generated, in_token.shape[1]))
        first_half_mask = self.slice(in_mask, (0, 0), (self.pics_generated, in_mask.shape[1]))

        _, channels, img_h, img_w = x_t.shape
        half = self.slice(x_t, (0, 0, 0, 0), (self.pics_generated, channels, img_h, img_w))
        combined = self.concat((half, half))

        last_half_tokens = self.broadcast_to(random_token)
        last_half_mask = self.broadcast_to(random_mask)
        tokens = self.concat((first_half_tokens, last_half_tokens))
        mask = self.concat((first_half_mask, last_half_mask))

        return combined, tokens, mask


class Guider(nn.Cell):
    def __init__(self, guidance_scale):
        super(Guider, self).__init__()
        # model attributes
        self.guidance_scale = guidance_scale

        # operations
        self.slice = mindspore.ops.Slice()
        self.concat = mindspore.ops.Concat(axis=0)
        self.concat_at_1 = mindspore.ops.Concat(axis=1)
        self.split = mindspore.ops.Split(axis=0, output_num=2)
        self.add = mindspore.ops.Add()
        self.mul = mindspore.ops.Mul()
        self.neg = mindspore.ops.Neg()

    '''
    x_t: tensor
    ts: tensor
    kwargs: dict, {tokens : num_of_pics*2 x 128 tensor, mask : num_of_pics*2 x 128 tensor}
    '''
    def construct(self, model_out):
        modelout_shape = model_out.shape
        eps = self.slice(model_out, (0, 0, 0, 0), (modelout_shape[0], 3, modelout_shape[2], modelout_shape[3]))
        rest = self.slice(model_out, (0, 3, 0, 0), (modelout_shape[0], 3, modelout_shape[2], modelout_shape[3]))

        cond_eps, uncond_eps = self.split(eps)

        diff_eps = self.add(cond_eps, self.neg(uncond_eps))
        scaled_diff_epq = self.mul(self.guidance_scale, diff_eps)
        half_eps = self.add(uncond_eps, scaled_diff_epq)
        eps = self.concat((half_eps, half_eps))
        out = self.concat_at_1((eps, rest))

        return out


class SamplingWithGuidance(nn.Cell):
    def __init__(self, model, guidance_scale, num_of_pics_generated):
        super(SamplingWithGuidance, self).__init__()
        self.combine_prompt = CombinePrompt(num_of_pics_generated)
        self.model = model
        self.guider = Guider(guidance_scale)
        self.broadcast_to = mindspore.ops.BroadcastTo((num_of_pics_generated * 2,))
        self.concat = mindspore.ops.Concat(axis=1)

    def construct(self, x_t, timesteps, in_token, in_mask, random_token, random_mask):
        combined, tokens, mask = self.combine_prompt(x_t, in_token, in_mask, random_token, random_mask)
        timesteps = self.broadcast_to(timesteps)
        model_out = self.model(combined, timesteps, tokens, mask)
        out = self.guider(model_out)
        return out
