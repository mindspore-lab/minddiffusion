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
"""
Various utilities for neural networks.
"""

import math

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore.common.initializer import initializer
from mindspore.nn.transformer.layers import _Linear


class SiLU(nn.Cell):
    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)
    

class GroupNorm32_swish1(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.cast = ops.Cast()
        self.silu = SiLU()
        self.swish = swish

    def construct(self, x):
        x = self.cast(x, ms.float32)
        y = super().construct(x)
        y = self.silu(y)
        y = self.cast(y, x.dtype)
        return y
    
    
class GroupNorm32_swish0(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.cast = ops.Cast()
        self.swish = swish

    def construct(self, x):
        x = self.cast(x, ms.float32)
        y = super().construct(x)
        y = self.cast(y, x.dtype)
        return y


def GroupNorm32(channels, swish=0.0):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if swish == 1.0:
        return GroupNorm32_swish1(num_channels=channels, num_groups=32, swish=swish)
    elif swish == 0.0:
        return GroupNorm32_swish0(num_channels=channels, num_groups=32, swish=swish)
    raise ValueError(f"unsupported swish: {swish}")


class ConvNd(nn.Cell):
    def __init__(self, dims, in_channels, out_channels, kernel_size, padding=0, stride=1, has_bias=False,
                 pad_mode='same', dtype=ms.float16):
        super().__init__()
        if dims == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                  padding=padding, has_bias=has_bias, pad_mode=pad_mode).to_float(dtype)
        elif dims == 2:  # Note the differences in Conv2d
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                  padding=padding, has_bias=has_bias, pad_mode=pad_mode).to_float(dtype)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                                  padding=padding, has_bias=has_bias, pad_mode=pad_mode).to_float(dtype)
        
    def construct(self, x, emb, encoder_out):
        x = self.conv(x)
        return x


def Linear(in_channel, out_channel, dtype=ms.float32):
    """
    Create a linear module.
    """
    return _Linear(in_channel, out_channel, compute_dtype=dtype)


def AvgPoolNd(dims, dtype=ms.float32, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs).to_float(dtype)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs).to_float(dtype)
    elif dims == 3:
        return ops.AvgPool3D(*args, **kwargs).to_float(dtype)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    weight = initializer("zeros", module.conv.weight.shape)
    bias_weight = initializer("zeros", module.conv.bias.shape)
    module.conv.weight.set_data(weight)
    module.conv.bias.set_data(bias_weight)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


class TimestepEmbedding(nn.Cell):
    def __init__(self, max_period=10000, dtype=ms.float16):
        super().__init__()
        self.exp = ops.Exp()
        self.concat = ops.Concat(axis=-1)
        self.cos = ops.Cos()
        self.sin = ops.Sin()
        self.log = ops.Log()
        self.dtype = dtype
        self.max_period = ms.Tensor(input_data=-math.log(max_period), dtype=dtype)
        self.cast = ops.Cast()
        self.print = ops.Print()

    def construct(self, timesteps, dim):
        half = dim // 2
        freqs = self.exp(self.max_period * ms.numpy.arange(start=0, stop=half, dtype=self.dtype) / half)
        args = timesteps[:, None] * freqs[None]
        embedding = self.concat((self.cos(args), self.sin(args)))
        embedding = self.cast(embedding, self.dtype)
        return embedding
