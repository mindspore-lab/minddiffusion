import numpy as np
from ddm import Unet, GaussianDiffusion, value_and_grad
from mindspore import Tensor

def test_one_step():
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 10,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )

    training_images = np.random.randn(1, 3, 128, 128).astype(np.float32) # images are normalized from 0 to 1
    noise = np.random.randn(1, 3, 128, 128).astype(np.float32)
    grad_fn = value_and_grad(diffusion, None, diffusion.trainable_params())

    loss, grads = grad_fn(Tensor(training_images), Tensor(noise), False)
    # after a lot of training

    sampled_images = diffusion.sample(batch_size = 1)
    print(sampled_images.shape) # (4, 3, 128, 128)

def test_one_step_self_condition():
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        self_condition=True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 10,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )

    training_images = np.random.randn(1, 3, 128, 128).astype(np.float32) # images are normalized from 0 to 1
    noise = np.random.randn(1, 3, 128, 128).astype(np.float32)
    grad_fn = value_and_grad(diffusion, None, diffusion.trainable_params())

    loss, grads = grad_fn(Tensor(training_images), Tensor(noise), True)
    # after a lot of training

    sampled_images = diffusion.sample(batch_size = 1)
    print(sampled_images.shape) # (4, 3, 128, 128)
