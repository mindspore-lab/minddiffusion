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
""" generator """
from collections import defaultdict
import numpy as np
from mindspore import Tensor

data_column = [
    'input_ids',
    'input_mask',
    'img',
    't',
    'weights'
]

data_column_supres = [
    'input_ids',
    'input_mask',
    'img',
    't',
    'weights',
    'low_res'
]

data_column_audio = [
    'input_ids',
    'position_ids',
    'attention_mask',
    'mel_targets',
    'duration_targets',
    'speakers',
    'texts',
    'src_lens',
    'mel_lens',
    'audio_max_text_len',
    'audio_max_mel_len',
    'pitch_targets',
    'energy_targets'
]

task2id = {
    'mlmThree': 0,
    'mrcThree': 1,
    'mrfrThree': 2,
    'mafrThree': 3,
    'macThree': 4,
    "itmThree": 5,
    'mrctThree': 6,
    "tdThree": 7,
    "idThree": 8,
    "adThree": 9,
    "ret": 10,
    "ftRet": 11
}


