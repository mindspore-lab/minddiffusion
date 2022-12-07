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
gaussian diffusion model with unet:
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.nn.probability.distribution as mds

from model.glide_text2im.gaussian_computation import *
from model.glide_text2im.model_creator import create_model, create_upsample_model
from model.glide_text2im.custom_types import LossType, ModelMeanType
from model.glide_text2im.model.gaussian_diffusion import QSampleAndMeans, PMeanVariance, PSampleDiffusionModel, PMeanVariance
from model.glide_text2im.losses import normal_kl,approx_standard_normal_cdf,discretized_gaussian_log_likelihood
from model.glide_text2im.diffusion_creator import create_gaussian_diffusion

class GaussianDiffusion(nn.Cell):
    def __init__(self, options, guidance_scale, shape, super_res=False, ckpt_path=None):
        super(GaussianDiffusion, self).__init__()
        self.shape = shape
        self.rand_like_uniformal = ms.ops.UniformReal()
        self.super_res = super_res
        if super_res:
            model=create_upsample_model(**options)
        else:
            model=create_model(**options)
        self.t2i_model = model
        self.pics_generated = int(shape[0] / 2)
        
        self.base_diffusion, betas = create_gaussian_diffusion(
            diffusion_steps=options["diffusion_steps"], noise_schedule=options["noise_schedule"],
            timestep_respacing=options["timestep_respacing"], class_balanced=options["class_balanced"],
            sketch_classes=options["sketch_classes"], guider_net=self.t2i_model,
            clip_denoised=True, denoised_net=None,  shape=shape, dtype=options["dtype"]
        )
        self.qsample = QSampleAndMeans(betas, shape)
        self.diffusion_with_p_sample = PSampleDiffusionModel(self.base_diffusion, shape=shape, dtype=options["dtype"])
        self.normal = mds.Normal(dtype=options["dtype"])
        self.cast = ms.ops.Cast()
        self.exp = ms.ops.Exp()
        self.add = ms.ops.Add()
        self.div = ms.ops.Div()
        self.mul = ms.ops.Mul()
        self.neg = ms.ops.Neg()
        self.pow = ms.ops.Pow()
        self.splits = ms.ops.Split(1, 2)
        self.concat_dim1 = ms.ops.Concat(axis=1)
        self.broadcast_to = ms.ops.BroadcastTo(shape)
        self.broadcast_to_2 = ms.ops.BroadcastTo((shape[0], ))
        self.slice = ms.ops.Slice()
        self.log = ms.ops.Log()
        self.prints = ms.ops.Print()
        self.log2 = ms.Tensor(2.0, ms.float32)
        self.vision_norm = ms.Tensor(1.0/255.0, dtype=options["dtype"])
        self.part1 = ms.Tensor(np.sqrt(2.0 / np.pi), dtype=options["dtype"])
        self.clip_value_min = ms.Tensor(1e-12,  ms.float32)
        self.standard_normal_rand = ms.ops.StandardNormal()
        self.reduce_sum = ms.ops.ReduceSum()

    def construct(self, tokens, mask, x_0, t, weigths, low_res=None, noise=None):
        noise = self.standard_normal_rand(x_0.shape)
        x_t, q_mean_variance = self.qsample(x_0, t, noise)
        if self.super_res:
            model_output = self.t2i_model(x_t, t, low_res, tokens, mask)
        else:
            model_output = self.t2i_model(x_t, t, tokens, mask)
            
        model_output, model_var_values = self.splits(model_output)
        model_output2 = ms.ops.stop_gradient(model_output)
        frozen_out = self.concat_dim1((model_output2, model_var_values))
        posterior_mean, _, posterior_log_variance_clipped = self.qsample.q_posterior_mean_variance(x_0, x_t, t)
        model_mean, _, model_log_variance, _, _, _, _ = self.diffusion_with_p_sample(x_t, t, tokens, mask, frozen_out=frozen_out)
        model_mean = self.cast(model_mean, ms.float32)
        model_log_variance = self.cast(model_log_variance, ms.float32)
        
        vb_loss = self.get_trainloss(
            x_0, 
            x_t, 
            t, 
            posterior_mean,
            posterior_log_variance_clipped,
            model_mean,
            model_log_variance,
            clip_denoised=False,
            weight=None)["output"]
        
        losses = self.pow((noise - model_output), 2)
        mse_loss = ms.numpy.mean(losses, axis=tuple(list(range(1, len(self.shape)))))
        loss = self.add(vb_loss, mse_loss)
        loss = loss.mean()
        
        return loss
    
    def get_trainloss(self, x_0, x_t, t, true_mean, true_log_variance_clipped, model_mean, model_log_variance, clip_denoised=True, weight=None):
        
        ## kl loss
        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, model_log_variance
        )
        if weight is not None:
            kl = self.mul(kl, weight)
        kl = self.div(ms.numpy.mean(kl, axis=tuple(list(range(1, len(self.shape))))), self.log(self.log2))
        decoder_nll = -self.discretized_gaussian_log_likelihood(
            x_0, means=model_mean, log_scales=0.5 * model_log_variance
        )
        if weight is not None:
            decoder_nll = self.mul(decoder_nll, weight)
        decoder_nll = self.div(ms.numpy.mean(decoder_nll, tuple(list(range(1, len(self.shape))))), self.log(self.log2))
        out = ms.numpy.where((t == 0), decoder_nll, kl)
        return {"output": out}

    def create_gaussian_diffusion(
        self,
        diffusion_steps,  # 1000
        noise_schedule,
        timestep_respacing,  # 200
        class_balanced,
        sketch_classes,
        guider_net=None,
        clip_denoised=True,
        denoised_net=None,
        shape=None,
        dtype=ms.float32
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
        timestep_map, new_betas = self.space_diffusion_from_base(use_timesteps, alphas_cumprod)

        diffusion = PMeanVariance(
            guider_net=guider_net, clip_denoised=clip_denoised, denoised_net=denoised_net, timestep_map=timestep_map,
            betas=betas, model_mean_type=ModelMeanType.EPSILON, loss_type=loss_type, sketch_classes=sketch_classes,
            shape=shape, dtype=dtype
        )
        return betas, diffusion

    def space_diffusion_from_base(self, use_timesteps, alphas_cumprod):
        timestep_map = []
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        return timestep_map, np.array(new_betas)


    def discretized_gaussian_log_likelihood(self, x, *, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.

        :param x: the target images. It is assumed that this was uint8 values,
                rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        #assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = self.exp(-log_scales)
        norm_plus_centered = self.add(centered_x, self.vision_norm)
        plus_in = self.mul(inv_stdv, norm_plus_centered)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        
        norm_minus_centered = centered_x-self.vision_norm
        min_in = self.mul(norm_minus_centered, inv_stdv)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = self.log(ms.ops.clip_by_value(cdf_plus, clip_value_min=self.clip_value_min))
        log_one_minus_cdf_min = self.log(ms.ops.clip_by_value(1.0 - cdf_min, clip_value_min=self.clip_value_min))
        cdf_delta = cdf_plus - cdf_min
        log_probs = ms.numpy.where(
            x < -0.999,
            log_cdf_plus,
            ms.numpy.where(x > 0.999, log_one_minus_cdf_min, self.log(ms.ops.clip_by_value(cdf_delta, clip_value_min=self.clip_value_min))),
        )
        return log_probs

    def approx_standard_normal_cdf(self, x):
        part2_2 = self.mul(self.pow(x, 3), 0.044715)
        part2 = self.add(x, part2_2)
        normal_cdf = self.mul(part2, self.part1)
        normal_cdf = 0.5 * (1.0 + ms.numpy.tanh(normal_cdf))
        return normal_cdf