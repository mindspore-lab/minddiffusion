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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from ldm.models.clip_zh.simple_tokenizer import tokenize
from .text_encoder import TextEncoder


class FrozenCLIPEmbedder_ZH(nn.Cell):
    def __init__(self, max_length=77, use_fp16=False):
        super(FrozenCLIPEmbedder_ZH, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.max_length = max_length
        self.tokenizer = tokenize
        self.transformer = TextEncoder(context_length=77, vocab_size=49408, output_dim=1024, width=1024, layers=23, heads=16, dtype=self.dtype)

    def tokenize(self, texts):
        return self.tokenizer(texts)

    def encode(self, text):
        batch_encoding = self.tokenize(text)
        outputs = self.transformer(batch_encoding)
        return outputs

    def construct(self, c):
        outputs = self.transformer(c)
        return outputs
