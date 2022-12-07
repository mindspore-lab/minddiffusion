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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .simple_nn import TimestepEmbedding, Linear
from .unet import UNetModel
from .xf import LayerNorm, Transformer


class Text2ImUNet(UNetModel):
    """
    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        n_vocab,
        *args,
        cache_text_emb=False,
        xf_ar=0.0,
        xf_padding=False,
        share_unemb=False,
        dtype=ms.float32,
        **kwargs,
    ):
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        super().__init__(*args, **kwargs, encoder_channels=xf_width, dtype=dtype)

        self.cast = ops.Cast()
        self.cat = ops.Concat(axis=1)
        self.print = ops.Print()
        self.transpose = ops.Transpose()
        self.timestep_embedding = TimestepEmbedding(dtype=dtype)

        self.transformer = Transformer(text_ctx, xf_width, xf_layers, xf_heads, dtype=dtype)
        if xf_final_ln:
            self.final_ln = LayerNorm([xf_width])
        else:
            self.final_ln = None

        self.token_embedding = nn.Embedding(n_vocab, xf_width).to_float(dtype)
        self.positional_embedding = ms.Parameter(ms.numpy.empty((text_ctx, xf_width), dtype=dtype))
        self.transformer_proj = Linear(xf_width, self.model_channels * 4, dtype=dtype)

        if self.xf_padding:
            self.padding_embedding = ms.Parameter(
                ms.numpy.empty((text_ctx, xf_width), dtype=dtype)
            )
        if self.xf_ar:
            self.unemb = Linear(xf_width, n_vocab, dtype=dtype)
            if share_unemb:
                self.unemb.weight = self.token_embedding.weight

        self.cache_text_emb = cache_text_emb

    def get_text_emb(self, tokens, mask):
        xf_in = self.token_embedding(tokens)
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            xf_in = ms.numpy.where(mask[..., None], xf_in, self.padding_embedding[None])
        xf_out = self.transformer(xf_in)
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = self.transpose(xf_out, (0, 2, 1))  # NLC -> NCL

        return xf_proj, xf_out

    def construct(self, x, timesteps, tokens=None, mask=None):
        hs = []
        emb = self.time_embed(self.timestep_embedding(timesteps, self.model_channels))
        xf_proj, xf_out = self.get_text_emb(tokens, mask)
        emb = emb + xf_proj
        h = x
        for celllist in self.input_blocks:
            for cell in celllist:
                h = cell(h, emb, xf_out)
            hs.append(h)

        for module in self.middle_block:
            h = module(h, emb, xf_out)
            
        i = -1
        for celllist in self.output_blocks:
            h = self.cat((h, hs[i]))
            for cell in celllist:
                h = cell(h, emb, xf_out)
            i -= 1
        h = self.out(h)
        h = self.out2(h, emb, xf_out)
        return h


class SuperResText2ImUNet(UNetModel):
    """
    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(
        self,
        image_size,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        n_vocab,
        *args,
        cache_text_emb=False,
        xf_ar=0.0,
        xf_padding=False,
        share_unemb=False,
        dtype=ms.float32,
        **kwargs,
    ):
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        super().__init__(*args, **kwargs, encoder_channels=xf_width, dtype=dtype)

        self.cast = ops.Cast()
        self.cat = ops.Concat(axis=1)
        self.add = ops.Add()
        self.round = ops.Round()
        self.div = ops.Div()
        self.neg = ops.Neg()
        self.mul = ops.Mul()
        self.print = ops.Print()
        self.transpose = ops.Transpose()
        self.timestep_embedding = TimestepEmbedding(dtype=dtype)
        self.resize_bilinear = ops.ResizeBilinear((image_size, image_size), align_corners=False)
        self.cat = ops.Concat(axis=1)

        self.transformer = Transformer(text_ctx, xf_width, xf_layers, xf_heads, dtype=dtype)
        if xf_final_ln:
            self.final_ln = LayerNorm([xf_width])
        else:
            self.final_ln = None

        self.token_embedding = nn.Embedding(n_vocab, xf_width).to_float(dtype)
        self.positional_embedding = ms.Parameter(ms.numpy.empty((text_ctx, xf_width), dtype=dtype))
        self.transformer_proj = Linear(xf_width, self.model_channels * 4, dtype=dtype)

        if self.xf_padding:
            self.padding_embedding = ms.Parameter(
                ms.numpy.empty((text_ctx, xf_width), dtype=dtype)
            )
        if self.xf_ar:
            self.unemb = Linear(xf_width, n_vocab, dtype=dtype)
            if share_unemb:
                self.unemb.weight = self.token_embedding.weight

        self.cache_text_emb = cache_text_emb
        self.const_1 = ms.Tensor(input_data=1, dtype=dtype)
        self.const_127_5 = ms.Tensor(input_data=127.5, dtype=dtype)

    def get_text_emb(self, tokens, mask):
        xf_in = self.token_embedding(tokens)
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            xf_in = ms.numpy.where(mask[..., None], xf_in, self.padding_embedding[None])
        xf_out = self.transformer(xf_in)
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = self.transpose(xf_out, (0, 2, 1))  # NLC -> NCL

        return xf_proj, xf_out



    def construct(self, x, timesteps, low_res=None, tokens=None, mask=None):
        upsample = self.resize_bilinear(low_res)
        x = self.cat((x, upsample))
        hs = []
        emb = self.time_embed(self.timestep_embedding(timesteps, self.model_channels))
        xf_proj, xf_out = self.get_text_emb(tokens, mask)
        emb = emb + xf_proj
        h = x
        for celllist in self.input_blocks:
            for cell in celllist:
                h = cell(h, emb, xf_out)
            hs.append(h)

        for module in self.middle_block:
            h = module(h, emb, xf_out)
            
        i = -1
        for celllist in self.output_blocks:
            h = self.cat((h, hs[i]))
            for cell in celllist:
                h = cell(h, emb, xf_out)
            i -= 1
        h = self.out(h)
        h = self.out2(h, emb, xf_out)
        return h
