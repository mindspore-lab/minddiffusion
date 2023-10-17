import mindspore
import numpy as np
from mindspore import Tensor
from ddm.layers import Conv2d, Conv2dV2

x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
conv_2d = Conv2d(120, 240, 4)

conv_2d_v2 = Conv2dV2(120, 240, 4)
conv_2d_v2.weight.set_data(conv_2d.weight.data)
conv_2d_v2.bias.set_data(conv_2d.bias.data)

output = conv_2d(x)
output2 = conv_2d_v2(x)
print(output, output2)

