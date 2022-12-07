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

class StopAtStep(ms.Callback):
    def __init__(self, start_step, stop_step, profiler):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = profiler
    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
    def end(self, run_context):
        self.profiler.analyse()

class StopAtEpoch(ms.Callback):
    def __init__(self, start_epoch, stop_epoch, profiler):
        super(StopAtEpoch, self).__init__()
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.profiler = profiler
    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.start_epoch:
            self.profiler.start()
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.stop_epoch:
            self.profiler.stop()
    def end(self, run_context):
        self.profiler.analyse()


##moxing callback
class UploadObs(ms.Callback):
    def __init__(self, ckpt_dir, upload_url, ckpt_prefix="") -> None:
        super(UploadObs, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.upload_url = upload_url
        self.ckpt_prefix = ckpt_prefix

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        print("cb_params", cb_params)
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        ckpt_name = self.ckpt_prefix + "-" + str(cur_epoch_num) + "_" + str(cur_step_in_epoch) + ".ckpt"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        moxing.file.copy(ckpt_path, self.upload_url)


class GetParametersEpoch(ms.Callback):
    def __init__(self) -> None:
        super(GetParametersEpoch, self).__init__()
        
    
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        train_net = cb_params.get("train_net")

class OverflowMonitor(ms.Callback):
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        overflow = cb_params.net_outputs[1]
        if overflow:
            print(f"overflow detected in epoch {cur_epoch_num} step {cur_step_in_epoch}")
        return super().step_end(run_context)