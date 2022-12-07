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
"""loader"""
import random
import time
from collections import defaultdict
from multiprocessing import Process
import numpy as np
from model.glide_text2im.train.data_loader import DataLoader
import model.glide_text2im.train.config as config
import mindspore as ms

# loss = self.concat((mlm_loss.view(1,), mafr_loss.view(1,), mrfr_loss.view(1,),
#                    mac_loss.view(1,), itm_loss.view(1,), td_loss.view(1,), id_loss.view(1,)))

task2id = {
    'mlmThree': 0,
    'mafrThree': 1,
    'mrfrThree': 2,
    'macThree': 3,
    "itmThree": 4,
    "tdThree": 5,
    "idThree": 6,
    "adThree": 7,
    "ret": 10,
    "ftRet": 11,
    "ftCap": 12,
    "vqa": 13,
    "ftT2I": 14,
}

class SupresMetaLoader():
    """ wraps multiple data loaders """

    def __init__(self, loaders, datalen, accum_steps=1, task_num=9):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.name2iter_copy = {}
        self.sampling_pools = []
        self.loaders = loaders
        self.datalen = datalen
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n] * r)

        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        self.step = 0
        self.accum_steps = accum_steps
        self.step_cnt = 0
        self.task_index_list = np.random.permutation(self.task_num)
        random.seed(1)
        self.all_ids = []

    def init_iter(self, task_name):
        self.name2iter[task_name] = iter(self.name2loader[task_name])

    def return_ids(self):
        return self.all_ids

    def get_batch_params(self, batch):
        """ get_batch_params """

        batch = defaultdict(lambda: None, batch)
        input_ids = batch.get('input_ids', None)
        input_mask = batch.get('input_mask', None)
        img = batch.get('img_feat', None)
        t = batch.get('t', None)
        weights = batch.get('weights', None)
        low_res = batch.get('low_res', None)

        return (input_ids, input_mask, img, t, weights, low_res)

    def get_batch_check(self, batch, input_ids, input_mask, img, t, weights,low_res):
        """ get_batch_check """

        self.bs = len(input_mask)  # add by zjzhao
        #         print("self.bs=========================", self.bs)
        # text
        if input_ids is None:
            input_ids = ms.Tensor(np.zeros((self.bs, config.MAX_FULL_TEXT_LEN)),dtype=ms.int32)
        # if position_ids is None:
        #     position_ids = ms.Tensor(np.zeros((1, config.MAX_FULL_TEXT_LEN)), dtype=ms.int32)
        if img is None:
            images = ms.Tensor(np.ones((self.bs, 3, config.IMG_PATCH_SIZE, config.IMG_PATCH_SIZE)),dtype=ms.float32)
        
        if t is None:
            t = ms.Tensor(((self.bs, config.MAX_TIME_STEPS)),dtype=ms.int32)
        
        if weights is None:
            weighs = ms.Tensor(np.ones((self.bs, config.MAX_TIME_STEPS)),dtype=ms.int32)
            
        if low_res is None:
            low_res = ms.Tensor(np.ones((self.bs, 3, config.IMG_PATCH_SIZE, config.IMG_PATCH_SIZE)),dtype=ms.float32)

        return (input_ids, input_mask, img, t, weights, low_res)

    def get_batch(self, batch, task):
        """ get_batch """

        (input_ids, input_mask, img, t, weights, low_res) = self.get_batch_params(batch)

        (input_ids, input_mask, img, t, weights, low_res) = self.get_batch_check(batch, input_ids, input_mask, img, t, weights, low_res)
        taskId = np.array([task2id[task]]).astype(np.int32)

        output = (input_ids, input_mask, img, t, weights, low_res)
        return output

    def __getitem__(self, index):
        start_time = time.time()
        if self.step_cnt == self.task_num:
            self.task_index_list = np.random.permutation(self.task_num)
            self.step_cnt = 0
        task_index = self.task_index_list[self.step_cnt]
        local_task = self.sampling_pools[task_index]

        iter_ = self.name2iter[local_task]

        name = local_task
        try:
            batch = next(iter_)
        except StopIteration:
            print("============EPOCH END=============", flush=True)
            self.init_iter(local_task)
            print("cost init iter time :", time.time() - start_time, flush=True)
            iter_ = self.name2iter[local_task]
            batch = next(iter_)


        task = name.split('_')[0]
        # Prepare Data
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        output = self.get_batch(batch, task)

        # if self.print_time:
        #     print("============index: {}, taskname: {}, costtime: {}".format(index, task, time.time() - start_time))
        self.step_cnt += 1
        return output

    def __len__(self):
        # return 180 216 11961(256)  47853 83745(128 300w)   1314(128 100000) 5672*9 3545*9
        # return 5672*9
        return self.datalen


class MetaLoader():
    """ wraps multiple data loaders """

    def __init__(self, loaders, datalen, accum_steps=1, task_num=9):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.name2iter_copy = {}
        self.sampling_pools = []
        self.loaders = loaders
        self.datalen = datalen
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n] * r)

        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        self.step = 0
        self.accum_steps = accum_steps
        self.step_cnt = 0
        self.task_index_list = np.random.permutation(self.task_num)
        random.seed(1)
        self.all_ids = []

    def init_iter(self, task_name):
        self.name2iter[task_name] = iter(self.name2loader[task_name])

    def return_ids(self):
        return self.all_ids

    def get_batch_params(self, batch):
        """ get_batch_params """

        batch = defaultdict(lambda: None, batch)
        input_ids = batch.get('input_ids', None)
        input_mask = batch.get('input_mask', None)
        img = batch.get('img_feat', None)
        t = batch.get('t', None)
        weights = batch.get('weights', None)
        
        return (input_ids, input_mask, img, t, weights)

    def get_batch_check(self, batch, input_ids, input_mask, img, t, weights):
        """ get_batch_check """

        self.bs = len(input_mask)  # add by zjzhao
        #         print("self.bs=========================", self.bs)
        # text
        if input_ids is None:
            input_ids = ms.Tensor(np.zeros((self.bs, config.MAX_FULL_TEXT_LEN)),dtype=ms.int32)
        # if position_ids is None:
        #     position_ids = ms.Tensor(np.zeros((1, config.MAX_FULL_TEXT_LEN)), dtype=ms.int32)
        if img is None:
            images = ms.Tensor(np.ones((self.bs, 3, config.IMG_PATCH_SIZE, config.IMG_PATCH_SIZE)),dtype=ms.float32)
        
        if t is None:
            t = ms.Tensor(((self.bs, config.MAX_TIME_STEPS)),dtype=ms.int32)
        
        if weights is None:
            weighs = ms.Tensor(np.ones((self.bs, config.MAX_TIME_STEPS)),dtype=ms.int32)

        return (input_ids, input_mask, img, t, weights)

    def get_batch(self, batch, task):
        """ get_batch """

        (input_ids, input_mask, img, t, weights) = self.get_batch_params(batch)

        (input_ids, input_mask, img, t, weights) = self.get_batch_check(batch, input_ids, input_mask, img, t, weights)
        
        taskId = np.array([task2id[task]]).astype(np.int32)
        output = (input_ids, input_mask, img, t, weights)
        return output

    def __getitem__(self, index):
        start_time = time.time()
        if self.step_cnt == self.task_num:
            self.task_index_list = np.random.permutation(self.task_num)
            self.step_cnt = 0
        task_index = self.task_index_list[self.step_cnt]
        local_task = self.sampling_pools[task_index]

        iter_ = self.name2iter[local_task]

        name = local_task
        try:
            batch = next(iter_)
        except StopIteration:
            print("============EPOCH END=============", flush=True)
            self.init_iter(local_task)
            print("cost init iter time :", time.time() - start_time, flush=True)
            iter_ = self.name2iter[local_task]
            batch = next(iter_)


        task = name.split('_')[0]
        # Prepare Data
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        output = self.get_batch(batch, task)

        # if self.print_time:
        #     print("============index: {}, taskname: {}, costtime: {}".format(index, task, time.time() - start_time))
        self.step_cnt += 1
        return output

    def __len__(self):

        return self.datalen

if __name__ == "__main__":
    print(config.__dict__)
