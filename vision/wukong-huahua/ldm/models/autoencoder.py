import mindspore
import mindspore.nn as nn

from ldm.modules.diffusionmodules.model import Decoder
from ldm.util import instantiate_from_config


class AutoencoderKL(nn.Cell):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_fp16=False
                 ):
        super().__init__()
        self.dtype = mindspore.float16 if use_fp16 else mindspore.float32
        self.image_key = image_key
        self.decoder = Decoder(dtype=self.dtype, **ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1, pad_mode="valid", has_bias=True).to_float(self.dtype)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1, pad_mode="valid", has_bias=True).to_float(self.dtype)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", mindspore.ops.standard_normal(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = mindspore.load_checkpoint(path)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        mindspore.load_param_into_net(self, sd, strict_load=False)
        print(f"Restored from {path}")

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
