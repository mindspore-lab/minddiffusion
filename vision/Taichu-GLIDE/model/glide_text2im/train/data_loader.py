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
""" dataloader """

import os


class DataLoader:
    """ DataLoader """

    def __init__(self, dataset, batch_sampler, collate_fn, is_train=True, device_num=256, drop_last=True):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collat_fn = collate_fn
        self.device_num = device_num
        rank_id_str = os.getenv('RANK_ID', '0')
        self.rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])  # 'RANK_ID': 'job24535502-job-facereidtome-hn-0/1'
        self.is_train = is_train
        self.drop_last = drop_last
        self.batch_size = len(next(iter(self.batch_sampler)))

    def __iter__(self):
        self.step_index = 0
        self.batch_indices = iter(self.batch_sampler)

        return self

    def __next__(self):
        
        if self.is_train:
            try:
                indices = next(self.batch_indices)
                if len(indices) != self.batch_size and self.drop_last:
                    return self.__next__()
            except StopIteration:
                self.batch_indices = iter(self.batch_sampler)
                indices = next(self.batch_indices)
            data = []
            per_batch = len(indices) // self.device_num
            index = indices[self.rank_id * per_batch:(self.rank_id + 1) * per_batch]
            for idx in index:
                data.append(self.dataset[idx])

            data = self.collat_fn(data)
            return data
        else:
            indices = next(self.batch_indices)
            data = []
            per_batch = len(indices) // self.device_num
            index = indices[self.rank_id * per_batch:(self.rank_id + 1) * per_batch]
            for idx in index:
                data.append(self.dataset[idx])

            data = self.collat_fn(data)

            return data


