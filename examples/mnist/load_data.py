

import numpy as np
import struct
import torch
import torchvision.transforms as transforms
import random

from array import array
from numpy import ndarray
from typing import Union


class MnistDataloader(object):
    def __init__(self, train_images_filepath: str, train_labels_filepath: str, eval_images_filepath: str, eval_labels_filepath: str):
        self.train_images_filepath = train_images_filepath
        self.train_labels_filepath = train_labels_filepath
        self.eval_images_filepath = eval_images_filepath
        self.eval_labels_filepath = eval_labels_filepath

    @staticmethod
    def read_images_labels(genomes: int, images_filepath: str, labels_filepath: str, max_records: int = None,
                           resize: tuple[int, int] = None, rand_rot=False, rand_flip=False, batch_size=64,
                           device=torch.device('cpu'), dtype=torch.float32):
        # Load images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images_: list[ndarray] = []
        for i in range(size):
            images_.append(
                [0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            # sys.exit(0)
            images_[i][:] = img
        images = torch.tensor(np.stack(images_), device='cpu', dtype=torch.float32).unsqueeze(1)

        # Load Labels
        labels_: list[int] = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels_ = array("B", file.read())
        labels = torch.tensor(labels_, device='cpu', dtype=torch.long)

        if max_records is not None:
            images, labels = images[:max_records], labels[:max_records]

        # Filter some transforms for specific labels
        def get_indices(exclusions: Union[int, list[int]]):
            if isinstance(exclusions, (int, float)):
                exclusions = [exclusions]
            return [x for x in range(len(labels)) if labels[x].item() not in exclusions]

        # Modify in batches
        def get_batches(indices_to_group: list[int], batch_size: int = None, shuffle=False):
            records = len(indices_to_group)
            if batch_size is None:
                batch_size = records
            assert records > 0
            if batch_size is None:
                batch_size = records
            else:
                assert batch_size > 0
                batch_size = min(batch_size, records)

            indices: list[int] = indices_to_group
            if shuffle:
                random.shuffle(indices)
            batch_indices: list[list[int]] = list()
            batch: list[int] = list()
            for i, index in enumerate(indices):
                batch.append(index)
                batch_is_filled = len(batch) == batch_size
                no_more_records = i == len(indices) - 1 and not batch_is_filled
                if batch_is_filled or no_more_records:
                    batch_indices.append(batch)
                    batch = list()

            return batch_indices

        # Resizing
        if resize is not None:
            transformation = transforms.Resize(resize)
            images = transformation(images)
        # Random Flips
        if rand_flip:
            for batch in get_batches(get_indices([2, 4, 5, 6, 7, 9]), batch_size, True):
                h_flip = transforms.RandomHorizontalFlip(p=0.5)
                v_flip = transforms.RandomVerticalFlip(p=0.5)
                images[batch] = v_flip(h_flip(images[batch]))
        # Random Rotations
        if rand_rot:
            rotations90 = [(-30, 30)]
            for batch in get_batches(get_indices([6, 9]), batch_size, True):
                images[batch] = transforms.RandomRotation(random.choice(rotations90))(images[batch])
            # rotations180 = [(0, 0), (180, 180)]
            # for batch in get_batches(get_indices([2, 4, 5, 6, 7, 9]), batch_size, True):
            #     images[batch] = transforms.RandomRotation(random.choice(rotations180))(images[batch])

        def genome_expansion(tensor: torch.Tensor):
            shape = tensor.shape
            return tensor.unsqueeze(0).expand(genomes, *shape)

        print(f"Loaded {len(images)} records.")
        return genome_expansion(images).to(device, dtype) / 255, genome_expansion(labels).to(device)

    def load_data(self, genomes: int, train_records: int = None, eval_records: int = None,
                  resize: tuple[int, int] = None, rand_rot=False, rand_flip=False, batch_size=64,
                  device=torch.device('cpu'), dtype=torch.float32):
        train_inp, train_out = self.read_images_labels(
            genomes, self.train_images_filepath, self.train_labels_filepath,
            train_records, resize, rand_rot, rand_flip, batch_size,
            device, dtype
        )
        eval_inp, eval_out = self.read_images_labels(
            genomes, self.eval_images_filepath, self.eval_labels_filepath,
            eval_records, resize, False, False, None,
            device, dtype
        )
        return (train_inp, train_out), (eval_inp, eval_out)
