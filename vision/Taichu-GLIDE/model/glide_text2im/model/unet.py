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
from abc import abstractmethod

import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from .simple_nn import AvgPoolNd, ConvNd, Linear, GroupNorm32, TimestepEmbedding, zero_module, SiLU


class Upsample(nn.Cell):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = ConvNd(dims, self.channels, self.out_channels, 3,
                               padding=1, has_bias=True, pad_mode='pad', dtype=dtype)

    def construct(self, x, emb, encoder_out):
        a, b, c, d = x.shape
        x = ops.ResizeNearestNeighbor((c * 2, d * 2))(x)
        return x


class Downsample(nn.Cell):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = ConvNd(dims, self.channels, self.out_channels, 3, stride=stride,
                             padding=1, has_bias=True, pad_mode='pad', dtype=dtype)
        else:
            assert self.channels == self.out_channels
            self.op = AvgPoolNd(dims, kernel_size=stride, stride=stride, dtype=dtype)

    def construct(self, x, emb, encoder_out):
        return self.op(x)


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()
        self.identity = ops.Identity()

    def construct(self, x):
        return self.identity(x)


class ResBlock(nn.Cell):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            dtype=ms.float32
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.ori_channels = channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.identity = Identity().to_float(dtype)
        self.split = ops.Split(1, 2)
        self.print = ops.Print()
        if self.dropout == 0.0:
            self.dropout_layer = self.identity
        else:
            self.dropout_layer = nn.Dropout(self.dropout).to_float(dtype)

        self.in_layers_0 = GroupNorm32(channels, swish=1.0)
        self.in_layers_1 = self.identity
        self.in_layers_2 = ConvNd(dims, channels, self.out_channels, 3, padding=1,
                                  has_bias=True, pad_mode='pad', dtype=dtype)

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype)
        else:
            self.h_upd = self.identity
            self.x_upd = self.identity
        self.emb_layers = nn.SequentialCell(
            SiLU().to_float(dtype),
            Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                dtype=dtype
            ),
        )

        self.out_layers_0 = GroupNorm32(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0)
        self.out_layers_1 = SiLU().to_float(dtype) if use_scale_shift_norm else self.identity
        self.out_layers_2 = self.dropout_layer
        self.out_layers_3 = zero_module(
            ConvNd(dims, self.out_channels, self.out_channels, 3,
                   padding=1, has_bias=True, pad_mode='pad', dtype=dtype)
        )

        if self.out_channels == channels:
            self.skip_connection = self.identity
        elif use_conv:
            self.skip_connection = ConvNd(dims, channels, self.out_channels, 3,
                                          padding=1, has_bias=True, pad_mode='pad', dtype=dtype)
        else:
            self.skip_connection = ConvNd(dims, channels, self.out_channels, 1, has_bias=True,
                                          pad_mode='pad', dtype=dtype)

    def construct(self, x, emb, encoder_out):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            h = self.in_layers_0(x)
            h = self.in_layers_1(h)
            h = self.h_upd(h, emb, encoder_out)
            x = self.x_upd(x, emb, encoder_out)
            h = self.in_layers_2(h, emb, encoder_out)
        else:
            h = self.in_layers_0(x)
            h = self.in_layers_1(h)
            h = self.in_layers_2(h, emb, encoder_out)

        emb_out = self.emb_layers(emb)
        if len(h.shape) - len(emb_out.shape) == 1:
            emb_out = emb_out[..., None]
        elif len(h.shape) - len(emb_out.shape) == 2:
            emb_out = emb_out[..., None, None]

        if self.use_scale_shift_norm:
            scale, shift = self.split(emb_out)
            h = self.out_layers_0(h) * (1 + scale) + shift
            h = self.out_layers_1(h)
            h = self.out_layers_2(h)
            h = self.out_layers_3(h, emb, encoder_out)
        else:
            h = h + emb_out
            h = self.out_layers_0(h)
            h = self.out_layers_1(h)
            h = self.out_layers_2(h)
            h = self.out_layers_3(h, emb, encoder_out)
        if self.out_channels == self.ori_channels:
            return self.identity(x) + h
        else:
            return self.skip_connection(x, emb, encoder_out) + h


class AttentionBlock(nn.Cell):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
            dtype=ms.float32
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = GroupNorm32(channels, swish=0.0)
        self.qkv = ConvNd(1, channels, channels * 3, 1, has_bias=True, pad_mode='pad', dtype=dtype)
        self.attention = QKVAttention(self.num_heads, dtype)
        if encoder_channels is not None:
            self.encoder_kv = ConvNd(1, encoder_channels, channels * 2, 1, has_bias=True, pad_mode='pad', dtype=dtype)
        self.proj_out = zero_module(
            ConvNd(1, channels, channels, 1, has_bias=True, pad_mode='pad', dtype=dtype)
        )

    def construct(self, x, emb, encoder_out=None):
        b, c, d, e = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1), emb, encoder_out)
        # need recheck
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out, emb, None)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h, emb, encoder_out)
        return x + h.reshape(b, c, d, e)


class QKVAttention(nn.Cell):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, dtype=ms.float32):
        super().__init__()
        self.n_heads = n_heads
        self.concat = ops.Concat(axis=-1)
        self.dtype = dtype

        self.sqrt = ops.Sqrt()
        self.softmax = nn.Softmax()
        self.print = ops.Print()
        self.split_1 = ops.Split(axis=1, output_num=3)
        self.split_2 = ops.Split(axis=1, output_num=2)
        self.cast = ops.Cast()
        self.transpose = ops.Transpose()

    def construct(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = self.split_1(qkv.reshape(bs * self.n_heads, ch * 3, length))
        # need recheck
        if encoder_kv is not None:
            ek, ev = self.split_2(encoder_kv.reshape(bs * self.n_heads, ch * 2, -1))
            k = self.concat((ek, k))
            v = self.concat((ev, v))
        ch = self.cast(ch, self.dtype)
        scale = 1 / self.sqrt(self.sqrt(ch))
        q = q * scale
        k = k * scale
        q = self.transpose(q, (0, 2, 1))
        weight = ops.matmul(q, k)  # More stable with f16 than dividing afterwards
        weight = self.cast(self.softmax(self.cast(weight, mindspore.float32)), weight.dtype)
        v = self.transpose(v, (0, 2, 1))
        a = self.transpose(ops.matmul(weight, v), (0, 2, 1))
        return a.reshape(bs, -1, length)


class UNetModel(nn.Cell):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            encoder_channels=None,
            dtype=ms.float16
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.cat = ops.Concat(axis=1)

        self.timestep_embedding = TimestepEmbedding()
        self.identity = Identity().to_float(dtype)
        time_embed_dim = model_channels * 4  # 768
        self.time_embed = nn.SequentialCell([
            Linear(model_channels, time_embed_dim, dtype=dtype),
            SiLU().to_float(dtype),
            Linear(time_embed_dim, time_embed_dim, dtype=dtype),
        ])
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim).to_float(dtype)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.CellList([
            nn.CellList([
                ConvNd(dims, in_channels, ch, 3, padding=1, has_bias=True, pad_mode='pad', dtype=dtype)
            ])
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            # input layers
            for _ in range(num_res_blocks):
                layers = nn.CellList()
                layers.append(ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=int(mult * model_channels),
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=dtype))
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                            dtype=dtype))
                self.input_blocks.append(layers)

                self._feature_size += ch
                input_block_chans.append(ch)

            # input channels
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.CellList([ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                        dtype=dtype
                    )])
                    if resblock_updown
                    else nn.CellList([Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=dtype)])
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # middle layers
        self.middle_block = nn.CellList([
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=dtype
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
                dtype=dtype
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=dtype
            ),
        ])

        self._feature_size += ch

        # output layers
        self.output_blocks = nn.CellList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = nn.CellList()
                layers.append(
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=dtype
                    )
                )
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                            dtype=dtype
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=dtype
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=dtype)
                    )
                    ds //= 2
                self.output_blocks.append(layers)
                self._feature_size += ch

        # head
        self.out = nn.SequentialCell([
            GroupNorm32(ch, swish=1.0),
            self.identity,
        ])
        self.out2 = zero_module(
            ConvNd(dims, input_ch, out_channels, 3, padding=1, has_bias=True, pad_mode='pad', dtype=dtype)
        )

    def construct(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(self.timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = self.cat((h, hs.pop()))
            h = module(h, emb)
        return self.out(h)
