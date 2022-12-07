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


def model_and_diffusion_defaults(
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        channel_mult=(1, 2, 3, 4),
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions=tuple([2, 4, 8]),
        dropout=0.9,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        n_vocab=50001,
        xf_padding=True,
        diffusion_steps=1000,
        noise_schedule="squaredcos_cap_v2",
        timestep_respacing="60",
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        cache_text_emb=False,
        inpaint=False,
        super_res=False,
        chinese=True,
        sketch=False,
        class_balanced=False,
        sketch_classes=0,
        dtype=mindspore.float32
):
    return dict(
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        n_vocab=n_vocab,
        xf_padding=xf_padding,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
        chinese=chinese,
        sketch=sketch,
        class_balanced=class_balanced,
        sketch_classes=sketch_classes,
        dtype=dtype
    )


def model_and_diffusion_upsample(
        image_size=256,
        num_channels=192,
        num_res_blocks=2,
        channel_mult=(1,1,2,2,4,4),
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions=tuple([32, 16, 8]),
        dropout=0.0,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        n_vocab=50257,
        xf_padding=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="fast27",
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        cache_text_emb=False,
        inpaint=False,
        super_res=False,
        chinese=False,
        sketch=False,
        class_balanced=False,
        sketch_classes=0,
        dtype=mindspore.float32
):
    return dict(
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        n_vocab=n_vocab,
        xf_padding=xf_padding,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
        chinese=chinese,
        sketch=sketch,
        class_balanced=class_balanced,
        sketch_classes=sketch_classes,
        dtype=dtype
    )
