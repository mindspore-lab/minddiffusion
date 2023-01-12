import math
import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal
from .layers import Upsample, Conv2d, Dense
from .ops import rsqrt, rearrange, softmax

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class BMM(nn.Cell):
    def __init__(self):
        super().__init__()
        self.bmm = ops.BatchMatMul()

    def construct(self, x, y):
        return self.bmm(x, y)

class Identity(nn.Cell):
    def construct(self, inputs):
        return inputs

class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def upsample(dim, dim_out = None):
    return nn.SequentialCell(
        Upsample(scale_factor = 2, mode = 'nearest'),
        Conv2d(dim, default(dim_out, dim), 3, padding = 1, pad_mode='pad')
    )

def downsample(dim, dim_out = None):
    return Conv2d(dim, default(dim_out, dim), 4, 2, 'pad', 1)

class WeightStandardizedConv2d(Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def construct(self, x):
        eps = 1e-5

        weight = self.weight
        mean = weight.mean((1, 2, 3), keep_dims=True)
        var = weight.var((1, 2, 3), keepdims=True)
        normalized_weight = (weight - mean) * rsqrt((var + eps))

        output = self.conv2d(x, normalized_weight.astype(x.dtype))
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output

class LayerNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.g = Parameter(initializer('ones', (1, dim, 1, 1)), name='g')

    def construct(self, x):
        eps = 1e-5
        var = x.var(1, keepdims=True)
        mean = x.mean(1, keep_dims=True)
        return (x - mean) * rsqrt((var + eps)) * self.g

class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def construct(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * - emb)
        self.emb = Tensor(emb, mindspore.float32)

    def construct(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = ops.concat((ops.sin(emb), ops.cos(emb)), axis=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Cell):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2

        self.weights = Parameter(initializer(Normal(1.0), (half_dim,)), name='weights',
                                 requires_grad = not is_random)
        self.pi = Tensor(math.pi, mindspore.float32)

    def construct(self, x):
        x = x.expand_dims(1)
        freqs = x * self.weights.expand_dims(0) * 2 * self.pi
        fouriered = ops.concat((ops.sin(freqs), ops.cos(freqs)), axis = -1)
        fouriered = ops.concat((x, fouriered), axis = -1)
        return fouriered

# building block modules

class Block(nn.Cell):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1, pad_mode='pad')
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def construct(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Cell):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.SequentialCell(
            nn.SiLU(),
            Dense(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = Conv2d(dim, dim_out, 1, pad_mode='valid') if dim != dim_out else Identity()

    def construct(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.expand_dims(-1).expand_dims(-1) 
            scale_shift = time_emb.split(axis=1, output_num=2)
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        h = h + self.res_conv(x)
        return h

class LinearAttention(nn.Cell):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1, pad_mode='valid', has_bias = False)

        self.to_out = nn.SequentialCell(
            Conv2d(hidden_dim, dim, 1, pad_mode='valid', has_bias = True),
            LayerNorm(dim)
        )

        self.map = ops.Map()
        self.partial = ops.Partial()
        self.bmm = BMM()

    def construct(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).split(1, 3)
        q, k, v = self.map(self.partial(rearrange, self.heads), qkv)

        q = softmax(q, -2)
        k = softmax(k, -1)

        q = q * self.scale
        v = v / (h * w)

        # 'b h d n, b h e n -> b h d e'
        context = self.bmm(k, v.swapaxes(2, 3))

        # 'b h d e, b h d n -> b h e n'
        # out = (context.expand_dims(-1) * q.expand_dims(-2)).sum(2)
        out = self.bmm(context.swapaxes(2, 3), q)

        out = out.reshape((b, -1, h, w))
        return self.to_out(out)

class Attention(nn.Cell):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1, pad_mode='valid', has_bias = False)
        self.to_out = Conv2d(hidden_dim, dim, 1, pad_mode='valid', has_bias = True)
        self.map = ops.Map()
        self.partial = ops.Partial()
        self.bmm = BMM()

    def construct(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).split(1, 3)
        q, k, v = self.map(self.partial(rearrange, self.heads), qkv)

        q = q * self.scale

        # 'b h d i, b h d j -> b h i j'
        # sim = (q.expand_dims(-1) * k.expand_dims(-2)).sum(2)
        sim = self.bmm(q.swapaxes(2, 3), k)
        attn = softmax(sim, axis=-1)
        # 'b h i j, b h d j -> b h i d'
        # out = (attn.expand_dims(3) * v.expand_dims(2)).sum(-1)
        out = self.bmm(attn, v.swapaxes(2, 3))
        # out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = out.swapaxes(-1, -2).reshape((b, -1, h, w))

        return self.to_out(out)
