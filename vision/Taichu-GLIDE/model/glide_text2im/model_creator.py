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

from model.glide_text2im.model.text2im_model import Text2ImUNet, SuperResText2ImUNet


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    n_vocab,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    chinese,
    sketch,
    class_balanced,
    sketch_classes,
    dtype):
    print("origin t2i net")
    net = Text2ImUNet(
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        n_vocab=n_vocab,
        xf_padding=xf_padding,
        in_channels=3,
        model_channels=num_channels,
        out_channels=6,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
        dtype=dtype
    )
    return net


def create_upsample_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    n_vocab,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    chinese,
    sketch,
    class_balanced,
    sketch_classes,
    dtype):
    print("super res net")
    net = SuperResText2ImUNet(
        image_size=image_size,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        n_vocab=n_vocab,
        xf_padding=xf_padding,
        in_channels=6,
        model_channels=num_channels,
        out_channels=6,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
        dtype=dtype
    )
    return net
