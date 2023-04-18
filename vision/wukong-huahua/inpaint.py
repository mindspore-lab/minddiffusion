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

import argparse
import datetime
import math
import os
import sys
import shutil

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import mindspore as ms
import mindspore.dataset.vision as vision
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
print("workspace:", workspace, flush=True)
sys.path.append(workspace)
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


def make_batch_sd(
        image,
        mask,
        txt,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = Tensor(image, dtype=mstype.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1   
    mask = Tensor(mask, dtype=mstype.float32)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": image.repeat(num_samples, axis=0),
        "txt": num_samples * [txt],
        "mask": mask.repeat(num_samples, axis=0),
        "masked_image": masked_image.repeat(num_samples, axis=0),
    }
    return batch

def inpaint(sampler, image, mask, prompt, seed, scale, sample_steps, num_samples=1, w=512, h=512):
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = Tensor(start_code, dtype=mstype.float32)

    batch = make_batch_sd(image, mask, txt=prompt, num_samples=num_samples)

    c = model.get_learned_conditioning(batch["txt"])

    c_cat = list()
    for ck in model.concat_keys:
        cc = batch[ck]
        if ck != model.masked_image_key:
            bchw = [num_samples, 4, h // 8, w // 8]
            cc = x = ops.ResizeNearestNeighbor((bchw[-2], bchw[-1]))(cc)
        else:
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
        c_cat.append(cc)
    c_cat = ops.concat(c_cat, axis=1)

    # cond
    cond = {"c_concat": c_cat, "c_crossattn": c}

    # uncond cond
    uc_cross = model.get_learned_conditioning(num_samples * [""])
    uc_full = {"c_concat": c_cat, "c_crossattn": uc_cross}

    shape = [model.channels, h // 8, w // 8]
    samples_cfg, intermediates = sampler.sample(
        sample_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=0.0,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc_full,
        x_T=start_code,
        x0=c_cat[:, 1:],
    )

    x_samples = model.decode_first_stage(samples_cfg)

    result = ops.clip_by_value((x_samples + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

    result = result.asnumpy().transpose(0, 2, 3, 1)
    result = result * 255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]

    return result


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def main(args):
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        device_target="Ascend",
        device_id=device_id,
        max_device_memory="30GB"
    )
    
    if args.save_graph:
        save_graphs_path = "graph"
        shutil.rmtree(save_graphs_path)
        ms.context.set_context(
            save_graphs=True,
            save_graphs_path=save_graphs_path
        )
    
    seed_everything(args.seed)

    if not os.path.isabs(args.config):
        args.config = os.path.join(workspace, args.config)
    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(config, f"{os.path.join(args.ckpt_path, args.ckpt_name)}")
    if args.sampler.lower() == "plms":
        sampler = PLMSSampler(model)
    else:
        raise TypeError("unsupported sampler type")
    
    img_size = args.img_size
    num_samples = args.num_samples
    prompt = args.prompt
    image = Image.open(args.img).convert("RGB")
    mask_image = Image.open(args.mask).convert("RGB")
    if args.aug == "resize":
        aug_func = lambda x_: x_.resize((img_size, img_size))
    elif args.aug == 'crop':
        assert img_size % 2 == 0
        mask_idx = np.where(np.array(mask_image)[:, :, 0] > 127.5)
        mask_center = np.array(list(map(np.mean, mask_idx)))[::-1].astype('int')
        mask_center = [x_.clip(img_size // 2, size_ - img_size // 2) for x_, size_ in zip(mask_center, image.size)]
        aug_func = lambda x_: x_.crop((mask_center[0] - img_size // 2, mask_center[1] - img_size // 2,
                                       mask_center[0] + img_size // 2, mask_center[1] + img_size // 2))
    elif args.aug == 'resizecrop':
        mask_idx = np.where(np.array(mask_image)[:, :, 0] > 127.5)
        mask_center = np.array(list(map(np.mean, mask_idx)))[::-1].astype('int')
        mask_range = max(*[x_.max() - x_.min() for x_ in mask_idx])
        new_img_size = math.ceil(mask_range / args.mask_ratio)
        mask_center = [x_.clip(new_img_size // 2, size_ - new_img_size // 2) for x_, size_ in
                       zip(mask_center, image.size)]
        aug_func = lambda x_: x_.crop((mask_center[0] - new_img_size // 2, mask_center[1] - new_img_size // 2,
                                       mask_center[0] + new_img_size // 2, mask_center[1] + new_img_size // 2)).resize(
            (img_size, img_size))
    else:
        aug_func = lambda x_: x_
    image = aug_func(image)
    mask_image = aug_func(mask_image)
    mask_image = Image.fromarray(np.array(mask_image)[:, :, -1] > 127.5)

    images = [image, mask_image]
    for _ in range(math.ceil(num_samples / args.batch_size)):
        output = inpaint(
            sampler=sampler,
            image=image,
            mask=mask_image,
            prompt=prompt,
            seed=args.seed,
            scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            num_samples=args.batch_size,
            h=img_size,
            w=img_size
        )
        images.extend(output)

    im_save = image_grid(images, 1, num_samples + 2)
    ct = datetime.datetime.now().strftime("%Y_%d_%b_%H_%M_%S_")
    img_name = ct + prompt.replace(" ", "_") + ".png"
    os.makedirs(args.save_path, exist_ok=True)
    im_save.save(os.path.join(args.save_path, img_name))
    print("finish inpaint.")


def seed_everything(seed):
    if seed:
        ms.set_seed(seed)
        np.random.seed(seed)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    if os.path.exists(ckpt):
        param_dict = ms.load_checkpoint(ckpt)
        if param_dict:
            param_not_load = ms.load_param_into_net(model, param_dict)
            print("param not load:", param_not_load)
    else:
        print(f"!!!Warning!!!: {ckpt} doesn't exist")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="path to origin image"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="path to mask image"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output",
        help="path to save image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wukong-huahua_inpaint_inference.yaml",
        help=""
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="models",
        help=""
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="wukong-huahua-inpaint-ms.ckpt",
        help=""
    )
    parser.add_argument(
        "--aug",
        type=str,
        default="resize",
        help="augment type"
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=.75,
        help=""
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="num of total samples"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help=""
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size of model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help=""
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=30,
        help=""
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="plms",
        help="support plms only"
    )
    parser.add_argument(
        "--save_graph",
        action='store_true',
        help=""
    )
    args = parser.parse_args()
    main(args)