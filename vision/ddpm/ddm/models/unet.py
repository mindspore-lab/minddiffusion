from functools import partial
from mindspore import nn, ops
from ..layers import Dense, Conv2d
from ..modules import default, ResnetBlock, RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb, \
    Residual, PreNorm, LinearAttention, Attention, downsample, upsample

class Unet(nn.Cell):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = Conv2d(input_channels, init_dim, 7, padding = 3, pad_mode='pad', has_bias=True)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.SequentialCell(
            sinu_pos_emb,
            Dense(fourier_dim, time_dim),
            nn.GELU(False),
            Dense(time_dim, time_dim)
        )

        # layers

        self.downs = nn.CellList([])
        self.ups = nn.CellList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.CellList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                downsample(dim_in, dim_out) if not is_last else Conv2d(dim_in, dim_out, 3, padding = 1, pad_mode='pad', has_bias=True)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.CellList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                upsample(dim_out, dim_in) if not is_last else  Conv2d(dim_out, dim_in, 3, padding = 1, pad_mode='pad')
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = Conv2d(dim, self.out_dim, 1, pad_mode='valid', has_bias=True)

    def construct(self, x, time, x_self_cond):
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = ops.zeros_like(x)
            x = ops.concat((x_self_cond, x), 1)
        x = self.init_conv(x)
        r = x.copy()
        t = self.time_mlp(time)
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        len_h = len(h) - 1
        for block1, block2, attn, upsample in self.ups:
            x = ops.concat((x, h[len_h]), 1)
            len_h -= 1
            x = block1(x, t)

            x = ops.concat((x, h[len_h]), 1)
            len_h -= 1
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = ops.concat((x, r), 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
