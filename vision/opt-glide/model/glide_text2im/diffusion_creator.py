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
from mindspore import load_checkpoint

from model.glide_text2im.gaussian_computation import *
from model.glide_text2im.model.gaussian_diffusion import GenerativePSampleDiffusionModel, DDimSampleDiffusionModel, PMeanVariance
from model.glide_text2im.custom_types import LossType, ModelMeanType
from model.glide_text2im.model.guider import SamplingWithGuidance
from model.glide_text2im.model_creator import create_model, create_upsample_model



def init_diffusion_model(options, guidance_scale, shape, ckpt_path=None):
    # init model
    model = create_model(**options)

    # init guidance
    pics_generated = int(shape[0] / 2)
    sampling_with_guidance = SamplingWithGuidance(model, guidance_scale, pics_generated)

    # init diffusion
    base_diffusion, _ = create_gaussian_diffusion(
        diffusion_steps=options["diffusion_steps"], noise_schedule=options["noise_schedule"],
        timestep_respacing=options["timestep_respacing"], class_balanced=options["class_balanced"],
        sketch_classes=options["sketch_classes"], guider_net=sampling_with_guidance,
        clip_denoised=True, denoised_net=None, dtype=options["dtype"], shape=shape
    )
    diffusion_with_p_sample = GenerativePSampleDiffusionModel(base_diffusion, shape=shape, dtype=options["dtype"])
    return diffusion_with_p_sample


def init_super_res_model(options, shape, ckpt_path=None):
    # init model
    up_sample_model = create_upsample_model(**options)

    if ckpt_path:
        load_checkpoint(ckpt_path, up_sample_model)

    # init diffusion
    base_diffusion, _ = create_gaussian_diffusion(
        diffusion_steps=options["diffusion_steps"], noise_schedule=options["noise_schedule"],
        timestep_respacing=options["timestep_respacing"], class_balanced=options["class_balanced"],
        sketch_classes=options["sketch_classes"], guider_net=up_sample_model,
        clip_denoised=True, denoised_net=None, dtype=options["dtype"], shape=shape
    )
    diffusion_with_ddim_sample = DDimSampleDiffusionModel(base_diffusion, shape=shape, dtype=options["dtype"])
    return diffusion_with_ddim_sample


def create_gaussian_diffusion(
    diffusion_steps,  # 1000
    noise_schedule,
    timestep_respacing,  # 200
    class_balanced,
    sketch_classes,
    guider_net=None,
    clip_denoised=True,
    denoised_net=None,
    shape=None,
    dtype=mindspore.float32
):
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  # 0-1之间，1000个数
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    if class_balanced:
        loss_type = LossType.BALANCED_MSE
    else:
        loss_type = LossType.MSE

    use_timesteps = space_timesteps(diffusion_steps, timestep_respacing)
    alphas_cumprod = alpha_calculator(betas)
    timestep_map, new_betas = space_diffusion_from_base(use_timesteps, alphas_cumprod)

    diffusion = PMeanVariance(
        guider_net=guider_net, clip_denoised=clip_denoised, denoised_net=denoised_net, timestep_map=timestep_map,
        betas=new_betas, model_mean_type=ModelMeanType.EPSILON, loss_type=loss_type, sketch_classes=sketch_classes,
        shape=shape, dtype=dtype
    )
    return diffusion, betas


def space_diffusion_from_base(use_timesteps, alphas_cumprod):
    timestep_map = []

    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    return timestep_map, np.array(new_betas)
