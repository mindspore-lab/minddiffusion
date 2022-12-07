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
import numpy as np
import mindspore.nn as nn
import mindspore.numpy as msnp

from model.glide_text2im.model.guider import SamplingWithGuidance
from model.glide_text2im.model.text2im_model import SuperResText2ImUNet,Text2ImUNet

class PSampleDiffusionModel(nn.Cell):
    def __init__(self, p_mean_variance_model, shape, dtype=mindspore.float32):
        super(PSampleDiffusionModel, self).__init__()
        
        self.p_mean_variance = p_mean_variance_model

    def construct(self, x, timesteps, token, mask, random_token=None, random_mask=None, is_train=True, frozen_out=None):
        p_mean_out = self.p_mean_variance(x=x, timesteps=timesteps, token=token, mask=mask,
                                          random_token=random_token, random_mask=random_mask,
                                          is_train=is_train, frozen_out=frozen_out)
        return p_mean_out

class GenerativePSampleDiffusionModel(nn.Cell):
    def __init__(self, p_mean_variance_model, shape, dtype=mindspore.float32):
        super(GenerativePSampleDiffusionModel, self).__init__()

        # sub models
        self.p_sample = PSample(shape=shape, dtype=dtype)
        self.p_mean_variance = p_mean_variance_model

    def construct(self, x, timesteps, token, mask, random_token=None, random_mask=None, is_train=False):
        p_mean_out = self.p_mean_variance(x=x, timesteps=timesteps, token=token, mask=mask,
                                          random_token=random_token, random_mask=random_mask,
                                          is_train=is_train)
        p_sample_out = self.p_sample(x, p_mean_out, timesteps)
        return p_sample_out

class DDimSampleDiffusionModel(nn.Cell):
    def __init__(self, p_mean_variance_model, shape, dtype=mindspore.float32):
        super(DDimSampleDiffusionModel, self).__init__()

        # sub models
        self.ddim_sample = DDimSample(shape=shape, dtype=dtype)
        self.p_mean_variance = p_mean_variance_model

    def construct(self, x, timesteps, token, mask, samples, is_train=False):
        p_mean_out = self.p_mean_variance(x=x, timesteps=timesteps, token=token, mask=mask, samples=samples, is_train=is_train)
        ddim_sample_out = self.ddim_sample(x, p_mean_out, timesteps)
        return ddim_sample_out


class PSample(nn.Cell):
    def __init__(self, shape, dtype=mindspore.float32):
        super(PSample, self).__init__()
        self.ones_mask = mindspore.ops.Ones()(shape, dtype)
        self.zeros_mask = mindspore.ops.Zeros()(shape, dtype)
        self.dtype = dtype

        # ops
        self.standard_normal_rand = mindspore.ops.StandardNormal()
        self.exp = mindspore.ops.Exp()
        self.mul = mindspore.ops.Mul()
        self.add = mindspore.ops.Add()
        self.cast = mindspore.ops.Cast()

    def construct(self, x, p_mean_variance_out, timesteps):
        mean, _, log_variance, pred_xstart, _, _, _ = p_mean_variance_out
        noise = self.standard_normal_rand(x.shape) 
        noise = self.cast(noise, self.dtype)
        scaled_guider = self.mul(0.5, log_variance)
        exp_guider = self.exp(scaled_guider)
        if timesteps == 0:
            masked_guider = self.mul(self.zeros_mask, exp_guider)
        else:
            masked_guider = self.mul(self.ones_mask, exp_guider)
        noised_guider = self.mul(masked_guider, noise)
        sample = self.add(mean, noised_guider)
        return sample, pred_xstart


class DDimSample(nn.Cell):
    def __init__(self, shape, dtype=mindspore.float32, eta=0.0):
        super(DDimSample, self).__init__()
        # attrs
        self.eta = mindspore.Tensor(input_data=eta, dtype=dtype)
        self.ones_mask = mindspore.ops.Ones()(shape, dtype)
        self.zeros_mask = mindspore.ops.Zeros()(shape, dtype)
        self.one = mindspore.Tensor(input_data=1, dtype=dtype)
        self.dtype = dtype

        # ops
        self.neg = mindspore.ops.Neg()
        self.sqrt = mindspore.ops.Sqrt()
        self.add = mindspore.ops.Add()
        self.mul = mindspore.ops.Mul()
        self.div = mindspore.ops.Div()
        self.cast = mindspore.ops.Cast()
        self.standard_normal_rand = mindspore.ops.StandardNormal()

    def construct(self, x, p_mean_variance_out, timesteps):
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        mean, _, log_variance, pred_xstart, eps, alpha_bar, alpha_bar_prev = p_mean_variance_out

        one_minus_alpha_bar = self.add(self.one, self.neg(alpha_bar))
        one_minus_alpha_bar_prev = self.add(self.one, self.neg(alpha_bar_prev))

        factor1 = self.sqrt(self.div(one_minus_alpha_bar_prev, one_minus_alpha_bar))
        factor2 = self.sqrt(self.div(one_minus_alpha_bar, alpha_bar_prev))
        sigma = self.mul(self.mul(self.eta, factor1), factor2)

        # Equation 12.
        square_sigma = self.mul(sigma, sigma)
        sqrt_comp_prev = self.sqrt(self.add(one_minus_alpha_bar_prev, self.neg(square_sigma)))
        eps_sqrt_comp_prev = self.mul(sqrt_comp_prev, eps)
        xstart_alpha_bar_prev = self.mul(pred_xstart, self.sqrt(alpha_bar_prev))

        mean_pred = self.add(eps_sqrt_comp_prev, xstart_alpha_bar_prev)

        noise = self.cast(self.standard_normal_rand(x.shape), self.dtype)
        sigma_noise = self.mul(sigma, noise)

        if timesteps == 0:
            masked_noise = self.mul(self.zeros_mask, sigma_noise)
        else:
            masked_noise = self.mul(self.ones_mask, sigma_noise)

        sample = self.add(mean_pred, masked_noise)
        return sample, pred_xstart


class PMeanVariance(nn.Cell):
    def __init__(self, guider_net, clip_denoised=None, denoised_net=None, timestep_map=None, betas=None,
                 model_mean_type=None, loss_type=None, sketch_classes=None, rescale_timesteps=False, shape=None, dtype=mindspore.float32):
        super(PMeanVariance, self).__init__()

        # attrs
        self.clip_denoised = clip_denoised
        self.denoised_net = denoised_net
        self.guider_net = guider_net
        self.shape = shape
        self.timestep_map = mindspore.Tensor(timestep_map, dtype)
        self.minus_one = mindspore.Tensor(-1)
        self.one = mindspore.Tensor(1)
        self.dtype = dtype
        print("type", type(self.guider_net))
        if isinstance(self.guider_net, SamplingWithGuidance):
            self.mode = "gen_res"
        elif isinstance(self.guider_net, SuperResText2ImUNet):
            self.mode = "sup_res"
        elif isinstance(self.guider_net, Text2ImUNet):
            self.mode = "train_mode"
        # ops
        self.cast = mindspore.ops.Cast()
        self.split = mindspore.ops.Split(axis=1, output_num=2)
        self.exp = mindspore.ops.Exp()
        self.add = mindspore.ops.Add()
        self.div = mindspore.ops.Div()
        self.mul = mindspore.ops.Mul()
        self.neg = mindspore.ops.Neg()
        self.clip_by_value = mindspore.ops.clip_by_value
        self.broadcast_to = mindspore.ops.BroadcastTo(shape)
        self.broadcast_to_2 = mindspore.ops.BroadcastTo((shape[0], ))
        self.broadcast_to_3 = mindspore.ops.BroadcastTo((shape[1], shape[2], shape[3]))
        #self.broadcast_to_4 = mindspore.ops.BroadcastTo((1, shape[1], shape[2], shape[3]))

        ##for broadcast
        self.expand_dims = mindspore.ops.ExpandDims()
        self.broadcast_to_12 = mindspore.ops.BroadcastTo((shape[0],shape[1]))
        self.broadcast_to_22 = mindspore.ops.BroadcastTo((shape[0],shape[1],shape[2]))
        self.broadcast_to_32 = mindspore.ops.BroadcastTo((shape[0],shape[1],shape[2],shape[3]))

        self.slice = mindspore.ops.Slice()
        self.log = mindspore.ops.Log()
        self.round = mindspore.ops.Round()
        self.const_1 = mindspore.Tensor(input_data=1, dtype=dtype)
        self.const_127_5 = mindspore.Tensor(input_data=127.5, dtype=dtype)

        # key diffusion param calculation: alpha/mean/variance
        assert len(np.array(betas.shape)) == 1, "betas must be 1-D"
        assert (np.array(betas) > 0).all() and (np.array(betas) <= 1).all()

        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.sketch_classes = sketch_classes
        self.rescale_timesteps = rescale_timesteps
        self.betas = np.array(betas, dtype=np.float32)  # Use float64 for accuracy.
        self.num_timesteps = int(self.betas.shape[0])
        self.alphas_cumprod = np.cumprod(1.0 - self.betas, axis=0)  # alpha = 1.0 - beta
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = mindspore.Tensor(np.sqrt(self.alphas_cumprod), dtype)
        self.sqrt_one_minus_alphas_cumprod = mindspore.Tensor(np.sqrt(1.0 - self.alphas_cumprod), dtype)
        self.log_one_minus_alphas_cumprod = mindspore.Tensor(np.log(1.0 - self.alphas_cumprod), dtype)
        self.sqrt_recip_alphas_cumprod = mindspore.Tensor(np.sqrt(1.0 / self.alphas_cumprod), dtype)
        self.sqrt_recipm1_alphas_cumprod = mindspore.Tensor(np.sqrt(1.0 / self.alphas_cumprod - 1), dtype)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = \
            mindspore.Tensor(np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])), dtype)
        self.posterior_mean_coef1 = \
            mindspore.Tensor(self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), dtype)
        self.posterior_mean_coef2 = \
            mindspore.Tensor((1.0 - self.alphas_cumprod_prev) * np.sqrt(1.0 - self.betas) / (1.0 - self.alphas_cumprod), dtype)
        self.betas_tensor = mindspore.Tensor(self.betas, dtype)
        self.alphas_cumprod_tensor = mindspore.Tensor(self.alphas_cumprod, dtype)  # alpha = 1.0 - beta
        self.alphas_cumprod_prev_tensor = mindspore.Tensor(self.alphas_cumprod_prev, dtype)
        self.zeros = mindspore.numpy.zeros((shape))
        self.print = mindspore.ops.Print()

    # samples, in_token, in_mask, random_token, random_mask
    def construct(self, x, timesteps, token, mask, samples=None, random_token=None, random_mask=None, is_train=True, frozen_out=None):
        min_log = self.extract_and_broadcast(self.posterior_log_variance_clipped, timesteps)
        max_log = self.extract_and_broadcast(self.log(self.betas_tensor), timesteps)
        mapped_timestep = self.broadcast_to_2(self.timestep_map[timesteps])
        model_output = frozen_out
        if not is_train:
            if self.mode == "gen_res":
                model_output = self.guider_net(x, mapped_timestep, token, mask, random_token, random_mask)
            elif self.mode == "sup_res":
                low_res = self.superResPreprocess(samples)
                model_output = self.guider_net(x, mapped_timestep, low_res, token, mask)
        model_output, model_var_values = self.split(model_output)
        frac = self.div(self.add(model_var_values, 1), 2)
        comp_frac = self.add(1, self.neg(frac))
        model_log_variance = self.add(frac*max_log, comp_frac * min_log)
        model_log_variance = self.cast(model_log_variance, self.dtype)
        model_variance = self.exp(model_log_variance)

        process_xstart = self._predict_xstart_from_eps(x_t=x, eps=model_output, timesteps=timesteps)
        pred_xstart = self.clip_by_value(process_xstart, self.minus_one, self.one)

        model_mean = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, timesteps=timesteps)
        eps = self._predict_eps_from_xstart(x, pred_xstart, timesteps)
        alpha_bar = self.extract_and_broadcast(self.alphas_cumprod_tensor, timesteps)
        alpha_bar_prev = self.extract_and_broadcast(self.alphas_cumprod_prev_tensor, timesteps)
        return model_mean, model_variance, model_log_variance, pred_xstart, eps, alpha_bar, alpha_bar_prev

    def q_posterior_mean_variance(self, x_start, x_t, timesteps):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        extracted_post_mean_coef1 = self.extract_and_broadcast(self.posterior_mean_coef1, timesteps)
        extracted_post_mean_coef2 = self.extract_and_broadcast(self.posterior_mean_coef2, timesteps)
        extracted_post_mean_coef1_start = extracted_post_mean_coef1 * x_start
        extracted_post_mean_coef2_t = extracted_post_mean_coef2 * x_t
        posterior_mean = self.add(extracted_post_mean_coef1_start, extracted_post_mean_coef2_t)

        return posterior_mean

    def _predict_xstart_from_eps(self, x_t, eps, timesteps):
        extracted_recip = self.extract_and_broadcast(self.sqrt_recip_alphas_cumprod, timesteps)
        extracted_recip1 = self.extract_and_broadcast(self.sqrt_recipm1_alphas_cumprod, timesteps)
        extracted_recip_xt = extracted_recip * x_t
        extracted_recip1_eps = self.neg(extracted_recip1 * eps)
        predict_out = self.add(extracted_recip_xt, extracted_recip1_eps)

        return predict_out

    def _predict_eps_from_xstart(self, x_t, pred_xstart, timesteps):
        extracted_recip = self.extract_and_broadcast(self.sqrt_recip_alphas_cumprod, timesteps)
        extracted_recip1 = self.extract_and_broadcast(self.sqrt_recipm1_alphas_cumprod, timesteps)
        extracted_recip_xt = extracted_recip * x_t
        extracted_recip_xstart = self.add(extracted_recip_xt, self.neg(pred_xstart))
        predict_out = extracted_recip_xstart/extracted_recip1
        return predict_out

    def extract_and_broadcast(self, arr, timesteps):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = msnp.choose(timesteps, arr)
        res = self.expand_dims(res, -1)
        res = self.broadcast_to_12(res)
        res = self.expand_dims(res, -1)
        res = self.broadcast_to_22(res)
        res = self.expand_dims(res, -1)
        res = self.broadcast_to_32(res)
        return res
    
    def superResPreprocess(self, samples):
        low_res = self.add(samples, self.const_1)
        low_res = self.mul(low_res, self.const_127_5)
        low_res = self.round(low_res)
        low_res = self.div(low_res, self.const_127_5)
        low_res = self.add(low_res, self.neg(self.const_1))
        return low_res


class QSampleAndMeans(nn.Cell):
    def __init__(self, betas, shape, dtype=mindspore.float32):
        super(QSampleAndMeans, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.betas = np.array(betas, dtype=np.float32)  # Use float64 for accuracy.
        self.num_timesteps = int(self.betas.shape[0])
        self.alphas_cumprod = np.cumprod(1.0 - self.betas, axis=0)  # alpha = 1.0 - beta
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.prints = mindspore.ops.Print()
        

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.one_minus_alpha_cimprod = mindspore.Tensor(1 - self.alphas_cumprod, dtype)
        self.sqrt_alphas_cumprod = mindspore.Tensor(np.sqrt(self.alphas_cumprod), dtype)
        self.sqrt_one_minus_alphas_cumprod = mindspore.Tensor(np.sqrt(1.0 - self.alphas_cumprod), dtype)
        self.log_one_minus_alphas_cumprod = mindspore.Tensor(np.log(1.0 - self.alphas_cumprod), dtype)
        self.sqrt_recip_alphas_cumprod = mindspore.Tensor(np.sqrt(1.0 / self.alphas_cumprod), dtype)
        self.sqrt_recipm1_alphas_cumprod = mindspore.Tensor(np.sqrt(1.0 / self.alphas_cumprod - 1), dtype)

         # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance_tensor = mindspore.ops.Tensor(self.posterior_variance, dtype)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = \
            mindspore.Tensor(np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])), dtype)
        self.posterior_mean_coef1 = \
            mindspore.Tensor(self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), dtype)
        self.posterior_mean_coef2 = \
            mindspore.Tensor((1.0 - self.alphas_cumprod_prev) * np.sqrt(1.0 - self.betas) / (1.0 - self.alphas_cumprod), dtype)
        self.betas_tensor = mindspore.Tensor(self.betas, dtype)
        self.alphas_cumprod_tensor = mindspore.Tensor(self.alphas_cumprod, dtype)  # alpha = 1.0 - beta
        self.alphas_cumprod_prev_tensor = mindspore.Tensor(self.alphas_cumprod_prev, dtype)

        self.rand_like_uniformal = mindspore.ops.UniformReal()
        self.standard_normal_rand = mindspore.ops.StandardNormal()
        #ops
        self.cast = mindspore.ops.Cast()
        self.split = mindspore.ops.Split(axis=1, output_num=2)
        self.exp = mindspore.ops.Exp()
        self.add = mindspore.ops.Add()
        self.div = mindspore.ops.Div()
        self.mul = mindspore.ops.Mul()
        self.neg = mindspore.ops.Neg()
        self.clip_by_value = mindspore.ops.clip_by_value
        self.broadcast_to = mindspore.ops.BroadcastTo(shape)
        self.broadcast_to_2 = mindspore.ops.BroadcastTo((shape[0], ))
        self.broadcast_to_3 = mindspore.ops.BroadcastTo((shape[1], shape[2], shape[3]))
        self.broadcast_to_4 = mindspore.ops.BroadcastTo((1, shape[1], shape[2], shape[3]))
        self.expand_dims = mindspore.ops.ExpandDims()
        self.broadcast_to_12 = mindspore.ops.BroadcastTo((shape[0],shape[1]))
        self.broadcast_to_22 = mindspore.ops.BroadcastTo((shape[0],shape[1],shape[2]))
        self.broadcast_to_32 = mindspore.ops.BroadcastTo((shape[0],shape[1],shape[2],shape[3]))
        self.slice = mindspore.ops.Slice()
        self.log = mindspore.ops.Log()
        self.print = mindspore.ops.Print()
        self.zeros = mindspore.numpy.zeros((shape))
    

    def construct(self, x_start, t, noise):
        x_t = self.qsample(x_start, t, noise)
        q_mean_variance = self.q_mean_variance(x_start, t)
        return x_t, q_mean_variance

    
    def qsample(self, x_start, t, noise):
        sqrt_alpha_xt = self.extract_and_broadcast(self.sqrt_alphas_cumprod, t) * x_start
        sqrt_minus_alpha_xt = self.extract_and_broadcast(self.sqrt_one_minus_alphas_cumprod, t) * noise
        res = self.add(sqrt_alpha_xt, sqrt_minus_alpha_xt)
        return res
    
    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = self.extract_and_broadcast(self.sqrt_alphas_cumprod, t) * x_start
        variance = self.extract_and_broadcast(self.one_minus_alpha_cimprod, t)
        log_variance = self.extract_and_broadcast(self.log_one_minus_alphas_cumprod, t)
        return mean, variance, log_variance
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        extracted_post_mean_coef1 = self.extract_and_broadcast(self.posterior_mean_coef1, t)
        extracted_post_mean_coef2 = self.extract_and_broadcast(self.posterior_mean_coef2, t)
        extracted_post_mean_coef1_start = extracted_post_mean_coef1 * x_start
        extracted_post_mean_coef2_t = extracted_post_mean_coef2 * x_t
        posterior_mean = self.add(extracted_post_mean_coef1_start, extracted_post_mean_coef2_t)
        posterior_variance = self.extract_and_broadcast(self.posterior_variance_tensor, t)
        posterior_log_variance_clipped = self.extract_and_broadcast(
            self.posterior_log_variance_clipped, t
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    
    def extract_and_broadcast(self, arr, timesteps):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = msnp.choose(timesteps, arr)
        res = self.expand_dims(res, -1)
        res = self.broadcast_to_12(res)
        res = self.expand_dims(res, -1)
        res = self.broadcast_to_22(res)
        res = self.expand_dims(res, -1)
        res = self.broadcast_to_32(res)
        return res
