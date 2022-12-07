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

import sys
sys.path.append("./")
import os
import re
import json
import argparse

import mindspore
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from model.glide_utils.img_utils import save_images
from model.glide_text2im.tokenizer.bpe import get_encoder
from model.glide_text2im.tokenizer.chinese_tokenizer import from_pretrained
from model.glide_text2im.tokenizer.caption_to_tokens import convert_input_to_token_gen, convert_input_to_token_super_res
from model.glide_text2im.default_options import model_and_diffusion_defaults, model_and_diffusion_upsample
from model.glide_text2im.diffusion_creator import init_diffusion_model, init_super_res_model
from model.glide_text2im.main_funcs import gaussian_p_sample_loop, ddim_sample_loop
from model.glide_text2im.model.srgan_util import SRGAN


def load_ckpt(net, ckpt_file, model_type="base"):
    if not ckpt_file:
        return
    print(f"start loading ckpt:{ckpt_file}")
    param_dict = load_checkpoint(ckpt_file)
    new_param_dict = {}
    for key,val in param_dict.items():
        keyL = key.split(".")
        new_keyL = []
        for para in keyL:
            if para == "diffusion_with_p_sample":
                continue
            else:
                new_keyL.append(para)
            if para == "guider_net" and model_type == "base":
                new_keyL.append("model")
        new_key = ".".join(new_keyL)
        new_param_dict[new_key] = val
    if param_dict:
        param_not_load = load_param_into_net(net, new_param_dict)
        print("param not load:", param_not_load)
    print(f"end loading ckpt:{ckpt_file}")
    
def read_prompts_file(file):
    prompts = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            prompts.append(txt)
    return prompts


def get_random_captions(file):
    captions = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            caption = line.strip().split('\t')[1]
            captions.append(caption)
    print(f"the num of all random captions: {len(captions)}")
    return captions
    

def main(args):
    mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_id=0)

    # hyper-params
    chinese = args.is_chinese
    denoise_steps = args.denoise_steps
    super_res_step = args.super_res_step
    guidance_scale = args.guidance_scale
    pics_generated = args.pics_generated
    tokenizer = from_pretrained(os.path.join(args.ckpt_path, args.tokenizer_model)) if chinese else get_encoder()
    ckpt_path_gen = os.path.join(args.ckpt_path, args.gen_ckpt)
    ckpt_path_super_res = os.path.join(args.ckpt_path, args.super_ckpt)
    ckpt_path_srgan = os.path.join(args.ckpt_path, args.srgan_ckpt)

    options = model_and_diffusion_defaults(timestep_respacing=str(denoise_steps), dtype=mindspore.float16)
    options_up = model_and_diffusion_upsample(dtype=mindspore.float16)
    input_shape = (pics_generated * 2, 3, options["image_size"], options["image_size"])
    up_shape = (pics_generated, 3, options_up["image_size"], options_up["image_size"])

    print("Initializing models...")
    # base model for 64*64 generative
    diffusion_model = init_diffusion_model(options=options, guidance_scale=guidance_scale, shape=input_shape, ckpt_path=ckpt_path_gen)
    load_ckpt(diffusion_model,ckpt_path_gen)

    # super res model 64*64 to 256*256
    super_res_model = init_super_res_model(options=options_up, shape=up_shape)
    load_ckpt(super_res_model,ckpt_path_super_res, model_type="supres")

    # super res model 256*256 to 1024*1024
    srgan = SRGAN(4, ckpt_path_srgan)
    
    print("read prompts_file...")
    prompts = read_prompts_file(args.prompts_file)
    for prompt in prompts:
        ori_image_path = os.path.join(args.output_path, prompt + ".jpg")
        upx4_image_path = os.path.join(args.output_path, prompt + "_up256.jpg")
        upx16_image_path = os.path.join(args.output_path, prompt + "_up1024.jpg")
        
        # Sample from the base model.
        token, mask = convert_input_to_token_gen(prompt, pics_generated, options['text_ctx'], tokenizer)
        samples = gaussian_p_sample_loop(diffusion_model=diffusion_model, token=token, mask=mask,
                                        shape=input_shape, num_timesteps=denoise_steps, tokenizer=tokenizer,
                                        text_ctx=options['text_ctx'], progress=True, dtype=options["dtype"], vocab_len = options["n_vocab"])[:pics_generated]
        save_images(samples, ori_image_path)

        token, mask = convert_input_to_token_super_res(prompt, pics_generated, options['text_ctx'], tokenizer)
        samples = ddim_sample_loop(super_res_model=super_res_model, samples=samples,
                                token=token, mask=mask, up_shape=up_shape, num_timesteps=super_res_step,
                                progress=True, dtype=options["dtype"])
        save_images(samples, upx4_image_path)
        
        samples = srgan.sr_handle(mindspore.ops.Cast()(samples, mindspore.float32))  # use fp32
        save_images(samples, upx16_image_path)

    result = {"infer_result": "success"}
    result = json.dumps(result)
    infe_result_path = os.path.join(args.output_path, "infer_result.json")
    with open(infe_result_path, 'w', encoding='utf-8') as f:
        f.write(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_chinese", default=True, type=bool, help="chinese or not")
    parser.add_argument("--denoise_steps", default=60, type=int, help="denoise steps")
    parser.add_argument("--super_res_step", default=27, type=int, help="super res step")
    parser.add_argument("--guidance_scale", default=5, type=int, help="guidance scale")
    parser.add_argument("--pics_generated", default=8, type=int, help="pic num for each prompt")
    
    parser.add_argument("--tokenizer_model", default="cog-pretrain.model", help="tokenizer model")
    parser.add_argument("--gen_ckpt", default="glide_gen.ckpt", help="gen ckpt")
    parser.add_argument("--super_ckpt", default="glide_super_res.ckpt", help="super ckpt")
    parser.add_argument("--srgan_ckpt", default="srgan.ckpt", help="srgan ckpt")
    
    parser.add_argument("--prompts_file", default="prompts.txt", help="prompts file")
    parser.add_argument("--data_path", default="", help="dataset path")
    parser.add_argument("--output_path", default="", help="output path")
    parser.add_argument("--ckpt_path",default="pretraind_models/", type=str, help="ckpt init path")
    parser.add_argument("--model_config_path", default="./configs/model_config.json", help="model_config")
    
    args = parser.parse_args()
    main(args)