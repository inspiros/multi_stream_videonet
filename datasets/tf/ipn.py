import math
import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

from ..utils.video_sampler import *

__all__ = ['IPNVideoDataGenerator', 'IPNFramesDataGenerator']


class IPNVideoDataGenerator(Sequence):
    def __init__(self,
                 video_dir,
                 annotation_file_path,
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 data_format='channels_first',
                 batch_size=1,
                 shuffle=True):
        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError(f'data_format must be either channels_first or channels_last, got {data_format}')
        self.video_dir = video_dir
        self.annotation_file_path = annotation_file_path
        self.sampler = sampler
        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

        self.data_format = data_format
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.clips = []
        self.labels = []
        with open(self.annotation_file_path) as f:
            lines = [_.strip().split(',') for _ in f.readlines()]
            for line in lines[1:]:
                video_file = line[0] + '.avi'
                label = int(line[2]) - 2
                if label < 0:
                    continue
                start_frame = int(line[3]) - 1
                end_frame = int(line[4]) - 1
                self.clips.append((video_file, start_frame, end_frame))
                self.labels.append(label)
        self.classes = np.unique(self.labels)

        self.indices = np.arange(len(self.clips))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.clips) / self.batch_size)

    def __getitem__(self, index):
        X, y = [], []
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        for item in inds:
            video_file, start_frame, end_frame = self.clips[item]
            video_file = os.path.join(self.video_dir, video_file)
            frames = self.sampler(video_file, start_frame, end_frame, sample_id=item)
            if self.to_rgb:
                frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            if self.transform is not None:
                frames = [self.transform(frame) if not self.use_albumentations
                          else self.transform(image=frame)['image'] for frame in frames]
            X.append(np.stack(frames))
        X = np.stack(X)
        if self.data_format == 'channels_first':
            X = X.transpose((0, 4, 1, 2, 3))
        y = np.array([self.labels[_] for _ in inds], dtype=np.int64)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.clips))
        if self.shuffle:
            np.random.shuffle(self.indices)

    @property
    def n_classes(self):
        return len(self.classes)


class IPNFramesDataGenerator(Sequence):
    def __init__(self,
                 frames_dir,
                 annotation_file_path,
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 data_format='channels_first',
                 batch_size=1,
                 shuffle=True):
        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError(f'data_format must be either channels_first or channels_last, got {data_format}')
        self.frames_dir = frames_dir
        self.annotation_file_path = annotation_file_path
        self.sampler = sampler
        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

        self.data_format = data_format
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.clips = []
        self.labels = []
        with open(self.annotation_file_path) as f:
            lines = [_.strip().split(',') for _ in f.readlines()]
            for line in lines[1:]:
                video = line[0]
                label = int(line[2]) - 2
                if label < 0:
                    continue
                start_frame = int(line[3]) - 1
                end_frame = int(line[4]) - 1
                self.clips.append((video, start_frame, end_frame))
                self.labels.append(label)
        self.classes = np.unique(self.labels)

        self.indices = np.arange(len(self.clips))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.clips) / self.batch_size)

    def __getitem__(self, index):
        X, y = [], []
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        for item in inds:
            video, start_frame, end_frame = self.clips[item]
            frames = self.sampler([os.path.join(self.frames_dir, video, f'{video}_{frame_id + 1:06d}.jpg')
                                   for frame_id in range(start_frame, end_frame + 1)], sample_id=item)
            if self.to_rgb:
                frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            if self.transform is not None:
                frames = [self.transform(frame) if not self.use_albumentations
                          else self.transform(image=frame)['image'] for frame in frames]
            X.append(np.stack(frames))
        X = np.stack(X)
        if self.data_format == 'channels_first':
            X = X.transpose((0, 4, 1, 2, 3))
        y = np.array([self.labels[_] for _ in inds], dtype=np.int64)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.clips))
        if self.shuffle:
            np.random.shuffle(self.indices)

    @property
    def n_classes(self):
        return len(self.classes)
