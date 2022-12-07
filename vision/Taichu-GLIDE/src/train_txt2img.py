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
import argparse
import json

from pathlib2 import Path
import mindspore as ms
import mindspore.nn as nn
from mindspore import Model
from mindspore.nn import DynamicLossScaleUpdateCell
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Profiler

from model.glide_utils.parallelConfig import ParallelConfig
from model.glide_utils.img_utils import save_images, read_image
from model.glide_utils.parallel_utils import _ClipByGlobalNorm
from model.glide_utils.learn_utils import LearningRate
from model.glide_utils.callbackConfig import StopAtEpoch,StopAtStep, UploadObs, GetParametersEpoch, OverflowMonitor
from model.glide_text2im.train import logger
from model.glide_text2im.train.image_datasets import load_data
from model.glide_text2im.tokenizer.bpe import get_encoder
from model.glide_text2im.train.cell_wrapper import ParallelTrainOneStepWithLossScaleCell, TrainOneStepWithLossScaleCell
from model.glide_text2im.tokenizer.chinese_tokenizer import from_pretrained
from model.glide_text2im.tokenizer.bpe import get_encoder
from model.glide_text2im.train.build_optimizer import build_optimizer
from model.glide_text2im.model.train_model import GaussianDiffusion
from model.glide_text2im.model_creation import add_dict_to_argparser

os.environ["OPENAI_LOGDIR"] = "./log"

def main(args, model_options):
    print("args",args)
    init()
    device_id = int(os.getenv('DEVICE_ID'))
    device_num = get_group_size()
    if args.use_parallel:
        device_id = int(os.getenv('DEVICE_ID'))
        print('device_id:{}'.format(device_id))
        if args.device_target != "local":
            print("start init")
            device_num = get_group_size()
            ParallelConfig.dp = device_num
            rank = get_rank()
            args.rank = rank
            print("device_id is {}, rank_id is {}, device_num is {}".format(
                device_id, rank, device_num))
            ms.context.reset_auto_parallel_context()
            ms.context.set_auto_parallel_context(
                parallel_mode=ms.context.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num)
        else:
            device_num = args.device_num
            rank = device_id % device_num
    
        ms.context.set_context(mode=ms.context.GRAPH_MODE)
    else:
        ms.context.set_context(mode=ms.context.GRAPH_MODE, device_id=device_id)
        
    logger.configure()
    logger.log("creating .")
    input_shape = (args.batch_size, 3, args.image_size, args.image_size)
    logger.log("training chinese:", args.is_chinese)
    logger.log("training sketch:", args.sketch)
    
    if args.is_chinese:
        token_path = os.path.join(args.pretrained_model_path, args.cog_model)
        tokenizer = from_pretrained(token_path)
        args.n_vocab = 50001
    else:
        tokenizer = get_encoder()
        args.n_vocab = 50257

    image_capth_file = os.path.join(args.data_path, args.image_caption_path_file)
    args.image_caption_path_file = image_capth_file

    dataset = load_data(
        image_caption_path_file=args.image_caption_path_file,
        timesteps=args.diffusion_steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        tokenizer=tokenizer,
        text_ctx=args.text_ctx,
        text_drop_p=args.text_drop_p,
        resolution_ori=args.resolution_ori if args.resolution_ori else args.image_size,
        using_data_sampler=args.using_data_sampler,
        is_super_res=args.is_super_res,
        device_num=device_num,
        device_id=device_id,
        data_path=args.data_path
        )

    ds = dataset.get_dataset_size()
    print(f"rank-{rank} batch num {ds}")
    
    train_net_loss = GaussianDiffusion(model_options, args.guidance_scale, shape=input_shape, super_res=args.is_super_res)
    if args.pretrained_model:
        init_ckpt_path = os.path.join(args.pretrained_model_path, args.pretrained_model)
        load_checkpoint(init_ckpt_path, train_net_loss)
        print("load ckpt finished")

    args.decay_steps = ds * args.epochs
    lr = LearningRate(args.start_learning_rate, args.end_learning_rate, args.warmup_steps, args.decay_steps)
    optimizer = build_optimizer(train_net_loss, args.optim, args.betas, lr)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=args.init_loss_scale,
                                             scale_factor=args.loss_scale_factor,
                                             scale_window=args.scale_window)
    if args.use_parallel:
        net_with_grads = ParallelTrainOneStepWithLossScaleCell(train_net_loss, optimizer=optimizer,
        scale_sense=update_cell, parallel_config=ParallelConfig)
    else:
        net_with_grads = TrainOneStepWithLossScaleCell(train_net_loss, optimizer=optimizer,
        scale_sense=update_cell)
        
    model = Model(net_with_grads)

    local_rank = 0
    new_epoch = args.epochs
    callback = [TimeMonitor(args.callback_size), LossMonitor(args.callback_size)]

    if args.save_summary:
        specified = {"collect_metric": True, "collect_graph": True, "collect_dataset_graph": True}
        summary_collector = SummaryCollector(summary_dir=os.path.join(args.output_path, 'summary'+str(device_id)),
        collect_specified_data=specified, collect_freq=1, keep_default_action=False, collect_tensor_freq=200)
        callback.append(summary_collector)
    
    if not args.save_checkpoint_steps:
        args.save_checkpoint_steps = ds
    if args.enable_mox:
        _file_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(_file_dir, args.output_path)
    else:
        ckpt_dir = os.path.join(args.output_path, "ckpt", f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_steps,
                                    keep_checkpoint_max=1,
                                    integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="OPT_glide",
                                    directory=ckpt_dir,
                                    config=config_ck)
    if get_rank()==0:
        callback.append(ckpoint_cb)
    
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if args.enable_mox:
        ckpt_prefix = "OPT_glide"
        mox_cb = UploadObs(ckpt_dir=ckpt_dir,upload_url=args.train_url,
                 ckpt_prefix=ckpt_prefix)
        
        callback.append(mox_cb)
        print("moxing ready")

    print("paramers call back get")
    get_params = GetParametersEpoch()
    callback.append(get_params)

    if args.enable_profiling:
        profile_path = args.profile_path
        profiler = Profiler(output_path=profile_path)
        if not args.dataset_sink_mode:
            profiler_stop_end_cb = StopAtStep(start_step=0, stop_step=40,
                                          profiler=profiler)
        else:
            profiler_stop_end_cb = StopAtEpoch(start_epoch=0, stop_epoch=args.epoch,
                                           profiler=profiler)
        callback += [profiler_stop_end_cb]
        print("profiling ready")

    print(f"rank-{rank} start_training...")
    
    model.train(
        new_epoch,
        dataset,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        sink_size=args.callback_size)


def load_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print(f"start loading ckpt:{ckpt_file}")
    param_dict = load_checkpoint(ckpt_file)
    new_param_dict = {}
    for key,val in param_dict.items():
        keyL = key.split(".")
        new_keyL = ["t2i_model"] + keyL
        new_key = ".".join(new_keyL)
        new_param_dict[new_key] = val
    print("new param dict", new_param_dict)
    if param_dict:
        param_not_load = load_param_into_net(net, new_param_dict)
        print("param not load:", param_not_load)
    print(f"end loading ckpt:{ckpt_file}")

def read_json(files):
    data = ""
    with open(files,'r',encoding = 'utf-8') as fp:
        data = json.load(fp)
    
    return data


if __name__ == "__main__":
    print('process id:', os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_chinese", default=True, type=bool, help="chinese or not")
    parser.add_argument("--is_super_res", default=False, type=bool, help="whether to train the generative model or super resolution")
    parser.add_argument("--resolution_ori", default=64, type=int, help="resolution_ori")
    parser.add_argument("--use_parallel", default=False, type=bool, help="use_parallel")
    parser.add_argument("--device_target", default="Communicate", help="local or Communicate")
    parser.add_argument("--device_num", default=1, type=int, help="device_num")
    parser.add_argument("--image_caption_path_file", default="image_caption_path_file.txt", help="dataset for train")
    parser.add_argument("--data_path", default="", help="dataset path")
    parser.add_argument("--schedule_sampler", default="uniform", help="noise sampler")
    parser.add_argument("--model_config", default="./utils/model_config.json", help="model_config")
    parser.add_argument("--guidance_scale", default=5, help="guidance_scale")
    parser.add_argument("--weight_decay", default=0.0, help="weight_decay")
    parser.add_argument("--batch_size", default=2, type=int, help="batch_size")
    parser.add_argument("--cog_model",default="", type=str, help="cog_model")
    parser.add_argument("--pretrained_model",default="", type=str, help="pretrained model name")
    parser.add_argument("--pretrained_model_path",default="pretraind_models/", type=str, help="ckpt init path")
    parser.add_argument("--text_drop_p", default=0.2, type=float, help="the unconditional generation scale during training")
    parser.add_argument("--using_data_sampler", default=False, help="")
    parser.add_argument("--start_learning_rate", default=1e-4, type=float, help="start learning rate")
    parser.add_argument("--end_learning_rate", default=1e-9, type=float, help="end learning rate")
    parser.add_argument("--warmup_steps", default=0, help="warmup steps")
    parser.add_argument("--epochs", default=10, type=int, help="epochs")
    parser.add_argument("--callback_size", default=1, type=int, help="callback size")
    parser.add_argument("--init_loss_scale",default=65366, type=int, help="loss scale for mixed precision")
    parser.add_argument("--loss_scale_factor", default=2, help="loss scale factor for mixed precision")
    parser.add_argument("--scale_window", default=1000, help="scale window")
    parser.add_argument("--save_summary", default=False, help="save summary")
    parser.add_argument("--output_path", default="", help="output path")
    parser.add_argument("--optim", default="adamw", help="optim")
    parser.add_argument("--betas", default=[0.9, 0.999], help="The scope of betas")
    parser.add_argument("--save_checkpoint_steps", default=10000, type=int, help="")
    parser.add_argument("--enable_profiling",default=False, help="enable profiling")
    parser.add_argument("--dataset_sink_mode",default=False, help="if use sink mode for training")
    parser.add_argument("--profile_path", default="./output", help="profile output path")
    parser.add_argument("--enable_mox", default=False, help="Whether to enable the obs return function")
    parser.add_argument("--train_url", default="", help="")
    parser.add_argument("--fp16", default=True,type=bool, help="use fp16 or not")
    
    args = parser.parse_args()
    model_option = read_json(args.model_config)
    model_option["channel_mult"] = tuple(model_option["channel_mult"])
    model_option["attention_resolutions"] = tuple(model_option["attention_resolutions"])
    model_option["dtype"] = ms.float32
    if args.fp16:
        model_option["dtype"] = ms.float16
    add_dict_to_argparser(parser, model_option)
    args = parser.parse_args()
    main(args, model_option)
    




