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
"""
sampler for length bucketing (batch by tokens)
"""
import random
import gc

class EasySampler:
    """
        Sampler for token bucket path
    """

    def __init__(self, dataset, batch_size, device_num=1):
        self.dataset = dataset
        self.per_batch = batch_size * device_num

    def _create_ids(self):
        return list(range(len(self.dataset)))

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)
        batches = [ids[i:i + self.per_batch] for i in range(0, len(ids) - self.per_batch, self.per_batch)]
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class BatchSampler:
    """
        Batch Sampler
    """

    def __init__(self, lens, batch_size, device_num):
        self._lens = lens
        self._batch_size = batch_size * device_num

    def _create_ids(self):
        return list(range(self._lens))

    def __iter__(self):
        ids = self._create_ids()
        batches = [ids[i:i + self._batch_size] for i in range(0, len(ids), self._batch_size)]
        gc.collect()
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")

