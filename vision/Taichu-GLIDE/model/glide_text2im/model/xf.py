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

import math

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops
import mindspore.ops as ops

from model.glide_text2im.model.simple_nn import Linear


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """
    def construct(self, x: ms.Tensor):
        y = super().construct(ops.Cast()(x, ms.float32))
        y = ops.Cast()(y, x.dtype)
        return y


class MultiheadAttention(nn.Cell):
    def __init__(self, n_ctx, width, heads, dtype):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = Linear(width, width * 3, dtype=dtype)
        self.c_proj = Linear(width, width, dtype=dtype)
        self.attention = QKVMultiheadAttention(width, heads, n_ctx, dtype)

    def construct(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class MLP(nn.Cell):
    def __init__(self, width, dtype):
        super().__init__()
        self.width = width
        self.c_fc = Linear(width, width * 4, dtype=dtype)
        self.c_proj = Linear(width * 4, width, dtype=dtype)
        self.gelu = nn.GELU()

    def construct(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Cell):
    def __init__(self, width: int, n_heads: int, n_ctx: int, dtype: mindspore.dtype):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx
        self.dtype = dtype
    
        self.concat = ops.Concat()
        self.sqrt = ops.Sqrt()
        self.softmax = nn.Softmax()
        self.print = ops.Print()
        self.split = ops.Split(axis=-1, output_num=3)
        self.cast = ops.Cast()
        self.transpose = ops.Transpose()

        self.scale = 1 / math.sqrt(math.sqrt(width * 3 // self.n_heads // 3))

    def construct(self, qkv):
        bs, _, _ = qkv.shape
        qkv = qkv.view(bs, self.n_ctx, self.n_heads, -1)
        q, k, v = self.split(qkv)
        q = q * self.scale
        k = k * self.scale
        q = self.transpose(q, (0, 2, 1, 3))
        k = self.transpose(k, (0, 2, 3, 1))
        weight = ops.matmul(q, k)
        wdtype = weight.dtype
        weight = self.cast(self.softmax(self.cast(weight, ms.float32)), wdtype)
        weight = self.transpose(weight, (0, 1, 2, 3))
        v = self.transpose(v, (0, 2, 1, 3))
        a = ops.matmul(weight, v)
        a = self.transpose(a, (0, 2, 1, 3))
        return a.reshape(bs, self.n_ctx, -1)


class ResidualAttentionBlock(nn.Cell):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        heads: int,
        dtype: mindspore.dtype
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
            dtype
        )
        self.ln_1 = LayerNorm([width])
        self.mlp = MLP(width, dtype)
        self.ln_2 = LayerNorm([width])

    def construct(self, x: ms.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Cell):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        dtype: mindspore.dtype,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.CellList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                    dtype
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x: ms.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

