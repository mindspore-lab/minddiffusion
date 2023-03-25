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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.function as F
import mindspore.ops.operations as P

class NoiseScheduleVP(nn.Cell):
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
    ):
        """
        Create a wrapper class for the forward SDE (VP type).
        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***
        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)
        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
            t = self.inverse_lambda(lambda_t)
        ===============================================================
        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).
        1. For discrete-time DPMs:
            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.
            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)
            Note that we always have alphas_cumprod = cumprod(betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.
            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).
        2. For continuous-time DPMs:
            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:
            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.
        ===============================================================
        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================
        """
        super().__init__()
        self.log = P.Log()
        self.cast = P.Cast()
        self.cos = P.Cos()
        self.sqrt = P.Sqrt()

        assert schedule in ["discrete"]

        if betas is not None:
            log_alphas = 0.5 * self.log(1 - betas).cumsum(dim=0)
        else:
            assert alphas_cumprod is not None
            log_alphas = 0.5 * self.log(alphas_cumprod)
        self.total_N = len(log_alphas)
        self.T = 1.
        self.t_array = F.linspace(ms.Tensor(0., ms.float32),
                                    ms.Tensor(1., ms.float32),
                                    self.total_N + 1)[1:].reshape((1, -1))
        self.log_alpha_array = log_alphas.reshape((1, -1,))

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return F.reshape(interpolate_fn(F.reshape(t, (-1, 1)), self.t_array, self.log_alpha_array), (-1,))

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return F.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return self.sqrt(1. - F.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * F.log(1. - F.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
    """
        log_alpha = -0.5 * \
            F.log(F.exp(P.Zeros()((1,))) + F.exp(-2. * lamb))
        t = interpolate_fn(log_alpha.reshape((-1, 1)), P.ReverseV2(axis=[1])(
            self.log_alpha_array), P.ReverseV2(axis=[1])(self.t_array))
        return t.reshape((-1,))


class DPM_Solver(nn.Cell):
    def __init__(self, model, steps=15, order=3, guidance_scale=1., lower_order_final=True, predict_x0=True, thresholding=False, max_val=1.):
        """Construct a DPM-Solver.
        We support both the noise prediction model ("predicting epsilon") and the data prediction model ("predicting x0").
        If `predict_x0` is False, we use the solver for the noise prediction model (DPM-Solver).
        If `predict_x0` is True, we use the solver for the data prediction model (DPM-Solver++).
            In such case, we further support the "dynamic thresholding" in [1] when `thresholding` is True.
            The "dynamic thresholding" can greatly improve the sample quality for pixel-space DPMs with large guidance scales.
        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            predict_x0: A `bool`. If true, use the data prediction model; else, use the noise prediction model.
            thresholding: A `bool`. Valid when `predict_x0` is True. Whether to use the "dynamic thresholding" in [1].
            max_val: A `float`. Valid when both `predict_x0` and `thresholding` are True. The max value for thresholding.

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        super().__init__()
        
        noise_schedule = NoiseScheduleVP('discrete', alphas_cumprod=model.alphas_cumprod)

        self.model = model
        self.noise_schedule = noise_schedule
        self.steps = steps
        self.order = order
        self.guidance_scale = guidance_scale
        self.lower_order_final = lower_order_final
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val
        self.cast = P.Cast()

    def get_model_input_time(self, t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        return (t_continuous - 1. / 1000.) * 1000.

    def noise_pred_fn(self, x, t_continuous, cond=None):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = F.broadcast_to(t_continuous, (x.shape[0],))
        t_input = self.get_model_input_time(t_continuous)
        output = self.model.apply_model(x, t_input, cond)
        return output

    def model_noise_prediction(self, x, t_continuous, unconditional_condition, condition):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = F.broadcast_to(t_continuous, (x.shape[0],))
    
        if self.guidance_scale == 1. or unconditional_condition is None:
            return self.noise_pred_fn(x, t_continuous, cond=condition)
        else:
            x_in = F.concat([x] * 2)
            t_in = F.concat([t_continuous] * 2)
            c_in = F.concat([unconditional_condition, condition])
            noise_uncond, noise = F.split(
                self.noise_pred_fn(x_in, t_in, cond=c_in), output_num=2)
            return noise_uncond + self.guidance_scale * (noise - noise_uncond)

    def data_prediction_fn(self, x, t, uc, c):
        """
        Return the data prediction model (with thresholding).
        """
        noise = self.model_noise_prediction(x, t, uc, c)
        dims = x.ndim
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(
            t), self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, dims) * noise) / \
            expand_dims(alpha_t, dims)
        if self.thresholding:
            p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
            temp = P.Sort(axis=1)(F.abs(x0).reshape((x0.shape[0], -1)))
            left_index = int((temp.shape[1]-1) * p)
            right_index = left_index + 1
            left_column = temp[:, left_index]
            right_column = temp[:, right_index]
            s = left_column + (right_column - left_column)*p
            s = expand_dims(F.maximum(
                s, self.max_val * F.ones_like(s)), dims)
            x0 = F.clip_by_value(x0, -s, s) / s
        return x0

    def model_fn(self, x, t, uc, c):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        return self.data_prediction_fn(x, t, uc, c)

    def get_time_steps(self, t_T, t_0, N):
        """Compute the intermediate time steps for sampling.
        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A tensor of the time steps, with the shape (N + 1,).
        """
        return F.linspace(ms.Tensor(t_T, ms.float32), ms.Tensor(t_0, ms.float32), N + 1)

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.
        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3, ] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3, ] * (K - 1) + [1]
            else:
                orders = [3, ] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2, ] * K
            else:
                K = steps // 2 + 1
                orders = [2, ] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1, ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(
                skip_type, t_T, t_0, K)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps)[
                P.CumSum()(ms.tensor([0, ] + orders), 0)]
        return timesteps_outer, orders

    def dpm_solver_first_update(self, x, s, t, model_s):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.
        Args:
            x: A tensor. The initial value at time `s`.
            s: A tensor. The starting time, with the shape (x.shape[0],).
            t: A tensor. The ending time, with the shape (x.shape[0],).
            model_s: A tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
        Returns:
            x_t: A tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.ndim
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = F.exp(log_alpha_t)

        phi_1 = F.expm1(-h)
        x_t = (
            expand_dims(sigma_t / sigma_s, dims) * x
            - expand_dims(alpha_t * phi_1, dims) * model_s
        )
        return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (x.shape[0],)
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.ndim
        model_prev_1, model_prev_0 = model_prev_list
        t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(
            t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = F.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = expand_dims(1. / r0, dims) * (model_prev_0 - model_prev_1)
        if self.predict_x0:
            x_t = (
                expand_dims(sigma_t / sigma_prev_0, dims) * x
                - expand_dims(alpha_t * (F.exp(-h) - 1.),
                              dims) * model_prev_0
                - 0.5 * expand_dims(alpha_t *
                                    (F.exp(-h) - 1.), dims) * D1_0
            )
        else:
            x_t = (
                expand_dims(
                    F.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                - expand_dims(sigma_t * (F.exp(h) - 1.),
                              dims) * model_prev_0
                - 0.5 * expand_dims(sigma_t *
                                    (F.exp(h) - 1.), dims) * D1_0
            )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (x.shape[0],)
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.ndim
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(
            t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = F.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = expand_dims(1. / r0, dims) * (model_prev_0 - model_prev_1)
        D1_1 = expand_dims(1. / r1, dims) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + expand_dims(r0 / (r0 + r1), dims) * (D1_0 - D1_1)
        D2 = expand_dims(1. / (r0 + r1), dims) * (D1_0 - D1_1)
        if self.predict_x0:
            x_t = (
                expand_dims(sigma_t / sigma_prev_0, dims) * x
                - expand_dims(alpha_t * (F.exp(-h) - 1.),
                              dims) * model_prev_0
                + expand_dims(alpha_t *
                              ((F.exp(-h) - 1.) / h + 1.), dims) * D1
                - expand_dims(alpha_t * ((F.exp(-h) - 1. +
                                          h) / h ** 2 - 0.5), dims) * D2
            )
        else:
            x_t = (
                expand_dims(F.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                - expand_dims(sigma_t * (F.exp(h) - 1.), dims) * model_prev_0
                - expand_dims(sigma_t * ((F.exp(h) - 1.) / h - 1.), dims) * D1
                - expand_dims(sigma_t * ((F.exp(h) - 1. - h) / h ** 2 - 0.5), dims) * D2
            )
        return x_t

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (x.shape[0],)
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t)
        else:
            raise ValueError(
                "Solver order must be 1 or 2 or 3, got {}".format(order))

    def sample(self, x, uc, c):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.
        =====================================================
        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver.
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.
        =====================================================
        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=False)
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `predict_x0 = True` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True)
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')
        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.
        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).
                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpm_solver` or `taylor`. We recommend `dpm_solver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.
        """
        t_0 = 1. / self.noise_schedule.total_N
        t_T = self.noise_schedule.T

        assert self.steps >= self.order
        timesteps = self.get_time_steps(t_T=t_T, t_0=t_0, N=self.steps)
        assert timesteps.shape[0] - 1 == self.steps

        vec_t = F.broadcast_to(timesteps[0], (x.shape[0],))
        model_prev_list = [self.model_fn(x, vec_t, uc, c)]
        t_prev_list = [vec_t]
        # Init the first `order` values by lower order multistep DPM-Solver.
        for init_order in range(1, self.order):
            vec_t = F.broadcast_to(timesteps[init_order], (x.shape[0],))
            x = self.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, vec_t, init_order)
            model_prev_list.append(self.model_fn(x, vec_t, uc, c))
            t_prev_list.append(vec_t)
        # Compute the remaining values by `order`-th order multistep DPM-Solver.
        for step in range(self.order, self.steps + 1):
            vec_t = F.broadcast_to(timesteps[step], (x.shape[0],))
            if self.lower_order_final and self.steps < 15:
                step_order = min(self.order, self.steps + 1 - step)
            else:
                step_order = self.order
            x = self.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, vec_t, step_order)
            for i in range(self.order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            t_prev_list[-1] = vec_t
            # We do not need to evaluate the final model value.
            if step < self.steps:
                model_prev_list[-1] = self.model_fn(x, vec_t, uc, c)
        return x

    def construct(self, x, uc, c):
        x = self.sample(x, uc, c)
        return x


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)
    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    expandd = P.ExpandDims()
    equal = P.Equal()
    gatherd = P.GatherD()
    cast = P.Cast()

    N, K = x.shape[0], xp.shape[1]
    all_x = F.concat((expandd(x, 2), F.tile(
        expandd(xp, 0), (N, 1, 1))), axis=2)
    sorted_all_x, x_indices = P.Sort(axis=2)(all_x)
    x_idx = P.Argmin(axis=2)(cast(x_indices, ms.float16))
    cand_start_idx = x_idx - 1

    start_idx = ms.numpy.where(
        equal(x_idx, 0),
        ms.Tensor(1),
        ms.numpy.where(
            equal(x_idx, K), ms.Tensor(K - 2), cand_start_idx,
        ),
    )
    end_idx = ms.numpy.where(
        equal(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = gatherd(sorted_all_x, 2, expandd(start_idx, 2)).squeeze(2)
    end_x = gatherd(sorted_all_x, 2, expandd(end_idx, 2)).squeeze(2)
    start_idx2 = ms.numpy.where(
        equal(x_idx, 0),
        ms.Tensor(0),
        ms.numpy.where(
            equal(x_idx, K), ms.Tensor(K - 2), cand_start_idx,
        ),
    )
    y_positions_expanded = F.broadcast_to(expandd(yp, 0), (N, -1, -1))
    start_y = gatherd(y_positions_expanded, 2,
                      expandd(start_idx2, 2)).squeeze(2)
    end_y = gatherd(y_positions_expanded, 2, expandd(
        (start_idx2 + 1), 2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.
    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]
