#!/bin/bash
# -*- coding: UTF-8 -*-
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

output_path=/glide/glide/output/
ckpt_path=/glide/pretraind_models/
model_config_path=/glide/configs/infer_model_config_glide.yaml
is_chinese=True
denoise_steps=60
super_res_step=27
pics_generated=4
tokenizer_model="cog-pretrain.model"
gen_ckpt="glide_gen.ckpt"
super_ckpt="glide_super_res.ckpt"
srgan_ckpt="srgan.ckpt"
prompts_file=./data/prompts.txt

python  src/txt2img.py \
        --output_path=$output_path \
        --ckpt_path=$ckpt_path \
        --model_config_path=$model_config_path \
        --is_chinese=$is_chinese \
        --denoise_steps=$denoise_steps \
        --super_res_step=$super_res_step \
        --pics_generated=$pics_generated \
        --tokenizer_model=$tokenizer_model \
        --gen_ckpt=$gen_ckpt \
        --super_ckpt=$super_ckpt \
        --srgan_ckpt=$srgan_ckpt \
        --prompts_file=$prompts_file \


