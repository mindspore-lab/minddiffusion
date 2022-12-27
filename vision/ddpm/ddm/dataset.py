from multiprocessing import cpu_count
from mindspore.dataset import ImageFolderDataset
from mindspore.dataset.vision import Resize, Inter, CenterCrop, ToTensor, RandomHorizontalFlip, Rescale
import numpy as np

def create_dataset(folder, image_size, exts = ['.jpg', '.jpeg', '.png', '.tiff'], \
                   augment_horizontal_flip=False, batch_size=32, num_shards=1, shard_id=0, \
                   shuffle=True, num_workers=cpu_count(), drop_remainder=False):
    num_workers = num_workers // num_shards
    dataset = ImageFolderDataset(folder, num_parallel_workers=num_workers, shuffle=False, \
                                 extensions=exts ,num_shards=num_shards, shard_id=shard_id, decode=True)
    transfroms = [
        Resize(image_size, Inter.BILINEAR),
        CenterCrop(image_size),
        ToTensor()
    ]

    dataset = dataset.project('image')
    if augment_horizontal_flip:
        transfroms.insert(1, RandomHorizontalFlip())

    dataset = dataset.map(transfroms, 'image')
    if shuffle:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset