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

import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore import context
import mindspore.numpy as np
import numpy


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    
    exp = ops.Exp()
    prints = ops.Print()
    pow = ops.Pow()
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + exp(logvar1 - logvar2)
        + (pow((mean1 - mean2), 2) * exp(-logvar2))
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    tanh = ops.Tanh()
    pow = ops.Pow()
    return 0.5 * (1.0 + tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    exp = ops.Exp()
    log = ops.Log()
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(ops.clip_by_value(cdf_plus, clip_value_min=1e-12, clip_value_max=1e10))
    log_one_minus_cdf_min = log(ops.clip_by_value((1.0 - cdf_min), clip_value_min=1e-12,
                                                  clip_value_max=1e10))
    cdf_delta = cdf_plus - cdf_min
    log_probs = np.where(
        x < -0.999,
        log_cdf_plus,
        np.where(x > 0.999, log_one_minus_cdf_min, log(ops.clip_by_value(
            cdf_delta, clip_value_min=1e-12, clip_value_max=1e10))),
    )
    assert log_probs.shape == x.shape
    return log_probs

if __name__ == "__main__":
    x = Tensor(numpy.random.standard_normal(10,).astype(numpy.float32))
    y = Tensor(numpy.random.standard_normal(10,).astype(numpy.float32))
    m1 = Tensor(numpy.random.standard_normal(10,).astype(numpy.float32))
    m2 = Tensor(numpy.random.standard_normal(10,).astype(numpy.float32))