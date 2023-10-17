import math
import numpy as np
from tqdm import tqdm
import mindspore
from mindspore import nn, ops, ms_function, Tensor
from ..modules import default
from ..ops import randn_like

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def extract(a, t, x_shape):
    # b = t.shape[0]
    # out = a.gather_elements(-1, t)
    # return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return a[t, None, None, None]

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps).astype(np.float32)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps).astype(np.float32)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Cell):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        alphas_cumprod_prev = np.pad(alphas_cumprod[:-1], (1, 0), constant_values = 1)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.sqrt_alphas_cumprod = Tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = Tensor(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = Tensor(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = Tensor(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = Tensor(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.posterior_variance = Tensor(posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.posterior_log_variance_clipped = Tensor(np.log(np.clip(posterior_variance, 1e-20, None)))
        self.posterior_mean_coef1 = Tensor(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = Tensor((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        p2_loss_weight = (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma
        self.p2_loss_weight = Tensor(p2_loss_weight)

        if self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss('none')
        elif self.loss_type == 'l2':
            self.loss_fn = nn.MSELoss('none')
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @ms_function
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)

        def maybe_clip(x, clip):
            if clip:
                return x.clip(-1., 1.)
            return x

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start, clip_x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start, clip_x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start, clip_x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            pred_noise = model_output
            x_start = model_output

        return pred_noise, x_start

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        _, x_start = self.model_predictions(x, t, x_self_cond)

        if clip_denoised:
            x_start.clip(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @ms_function
    def p_sample(self, x, t, x_self_cond = None, clip_denoised = True):
        batched_times = ops.ones((x.shape[0],), mindspore.int32) * t
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = randn_like(x) if t > 0 else ops.zeros_like(x) # no noise if t == 0
        pred_img = model_mean + ops.exp(0.5 * model_log_variance) * noise
        return pred_img, x_start

    def p_sample_loop(self, shape):
        img = np.random.randn(*shape).astype(np.float32)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            x_start = Tensor(x_start) if x_start is not None else x_start
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(Tensor(img), Tensor(t, mindspore.int32), self_cond)
            img, x_start = img.asnumpy(), x_start.asnumpy()

        img = unnormalize_to_zero_to_one(img)
        return img

    def ddim_sample(self, shape, clip_denoised = True):
        batch, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = np.linspace(-1, total_timesteps - 1, sampling_timesteps + 1).astype(np.int32)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = np.random.randn(*shape).astype(np.float32)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # time_cond = ops.fill(mindspore.int32, (batch,), time)
            time_cond = np.full((batch,), time).astype(np.int32)
            x_start = Tensor(x_start) if x_start is not None else x_start
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(Tensor(img, mindspore.float32), Tensor(time_cond), self_cond, clip_denoised)
            pred_noise, x_start = pred_noise.asnumpy(), x_start.asnumpy()
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * np.sqrt(((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)))
            c = np.sqrt(1 - alpha_next - sigma ** 2)

            noise = np.random.randn(*img.shape)

            img = x_start * np.sqrt(alpha_next) + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)

        return img

    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b = x1.shape[0]
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = ops.stack([mindspore.Tensor(t)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, ops.fill(mindspore.int32, (b,), i))

        return img

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise, random_cond):
        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        if self.self_condition:
            if random_cond:
                _, x_self_cond = self.model_predictions(x, t)
                x_self_cond = ops.stop_gradient(x_self_cond)
            else:
                x_self_cond = ops.zeros_like(x)
        else:
            x_self_cond = ops.zeros_like(x)
        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            target = noise
        # else:
        #     raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.reshape(loss.shape[0], -1)
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def construct(self, img, t, noise, random_cond):
        # b = img.shape[0]
        # t = randint(0, self.num_timesteps, (b,))

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, noise, random_cond)
