import math
import pathlib
import numpy as np
from PIL import Image
from typing import BinaryIO, Union, Optional, Tuple

def make_grid(
    tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0):

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = np.stack(tensor, axis=0)

    if tensor.ndim == 2:  # single image H x W
        tensor = tensor.expand_dims(0)
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = np.concatenate((tensor, tensor, tensor), 0)
        tensor = tensor.expand_dims(0)

    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = np.concatenate((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor_list = []  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(
                value_range, tuple
            ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img = np.clip(img, low, high)
            img = (img - low) / (max(high - low, 1e-5))
            return img

        def norm_range(t, value_range):
            if value_range is not None:
                return norm_ip(t, value_range[0], value_range[1])
            else:
                return norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                tensor_list.append(norm_range(t, value_range))
        else:
            tensor_list = norm_range(tensor, value_range)

        if isinstance(tensor_list, np.ndarray):
            tensor = tensor_list
        else:
            tensor = np.concatenate(tensor_list)

    assert isinstance(tensor, np.ndarray)
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = np.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding : (y + 1) * height, x * width + padding: (x + 1) * width] = tensor[k]
            k = k + 1
    return grid

def to_image(tensor,
             fp: Union[str, pathlib.Path, BinaryIO],
             format=None,
             **kwargs):
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid * 255 + 0.5
    ndarr = np.clip(ndarr, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)