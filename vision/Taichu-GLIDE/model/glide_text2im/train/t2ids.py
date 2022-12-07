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
TextToImage Datasets
"""
from toolz.sandbox import unzip
import os
import json
import numpy as np
import mindspore.dataset.vision.c_transforms as C
from mindspore.dataset.vision.utils import Inter
from PIL import Image


def pad_tensors(tensors, lens=None, pad=0, max_len=50):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.shape[0] for t in tensors]
    if max_len == -1:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[-1]
    dtype = tensors[0].dtype
    output = np.zeros((bs, max_len, hid), dtype=dtype)
    if pad:
        output.fill(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output[i, :l, ...] = t
    return output

def pad_tensors_pos(tensors, lens, feat, max_len=50):
    """ pad_tensors_pos """
    if tensors is None or tensors[0] is None:
        return np.expand_dims(np.arange(0, feat.shape[1], dtype=np.int64), 0)
    return pad_tensors(tensors, lens, max_len=max_len)

def get_ids_three(ids_path):
    ids = json.load(open(ids_path))
    size, rank = get_size_rank()
    return ids[rank::size]

def get_size_rank():
    size, rank = 1, 0
    return size, rank

def pad_sequence(sequences, batch_first=True, padding_value=0.0, max_lens=50):
    """pad_sequence"""
    lens = [len(x) for x in sequences]
    if max_lens == -1:
        max_lens = max(lens)

    padded_seq = []
    for x in sequences:
        pad_width = [(0, max_lens - len(x))]
        padded_seq.append(np.pad(x, pad_width, constant_values=(padding_value, padding_value)))

    sequences = np.stack(padded_seq, axis=0 if batch_first else 1)
    return sequences


def pad_sequence_(sequences, batch_first=False, padding_value=0.0, max_lens=50):
    """pad_sequence"""
    if sequences[0] is None:
        return None
    return pad_sequence(sequences, batch_first, padding_value, max_lens)

def t2i_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    :audio_feat   (n, audio_size, audio_dim)
    """
    img_feat, input_ids, input_mask, t, weights = map(list, unzip(inputs))

 
    batch = {
        'input_ids': input_ids,
        #'position_ids': position_ids,
        'input_mask': input_mask,
        'img_feat': img_feat,
        't': t,
        'weights': weights
    }
    return batch

def t2i_collate_supres(inputs):
    """
    Return:
    datas
    """
    img_feat, input_ids, input_mask, t, weights, low_res = map(list, unzip(inputs))
    

 
    batch = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'img_feat': img_feat,
        't': t,
        'weights': weights,
        'low_res': low_res,
    }
    return batch