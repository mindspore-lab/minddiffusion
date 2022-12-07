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

import model.glide_text2im.train.image_datasets as data_reader


def convert_input_to_token_gen(input_line, pics_generated, text_ctx, tokenizer):
    tokens, mask = encode_and_pad(input_line, text_ctx, tokenizer)
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], text_ctx)
    return (
        mindspore.Tensor([tokens] * pics_generated + [uncond_tokens] * pics_generated, dtype=mindspore.int32),
        mindspore.Tensor([mask] * pics_generated + [uncond_mask] * pics_generated, dtype=mindspore.int32)
    )


def convert_input_to_token_super_res(input_line, pics_generated, text_ctx, tokenizer):
    tokens, mask = encode_and_pad(input_line, text_ctx, tokenizer)
    tokens = mindspore.Tensor([tokens] * pics_generated, dtype=mindspore.int32)
    mask = mindspore.Tensor([mask] * pics_generated, dtype=mindspore.int32)
    return tokens, mask


def encode_and_pad(input_line, text_ctx, tokenizer):
    # Pack the tokens together into model kwargs.
    tokens = tokenizer.encode(input_line)
    tokens, mask = tokenizer.padded_tokens_and_mask(tokens, text_ctx)  # text_ctx 128
    return tokens, mask
