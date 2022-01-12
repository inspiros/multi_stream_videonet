import math
import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

from ..utils.video_sampler import *

__all__ = ['AFORSVideoDataGenerator']


class AFORSVideoDataGenerator(Sequence):
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
            subject_dirs = [_.strip() for _ in f.readlines()]
            for subject_dir in subject_dirs:
                subject_dir_path = os.path.join(self.video_dir, subject_dir)
                for timestamp_dir in sorted(os.listdir(subject_dir_path)):
                    timestamp_dir_path = os.path.join(subject_dir_path, timestamp_dir)
                    for video_file in filter(lambda _: _.endswith('.mp4'),
                                             sorted(os.listdir(timestamp_dir_path))):
                        label = int(os.path.splitext(video_file)[0]) - 1
                        video_file = os.path.join(subject_dir, timestamp_dir, video_file)
                        self.clips.append((video_file, subject_dir))
                        self.labels.append(label)
        self.classes = np.unique(self.labels)

        self.indices = np.arange(len(self.clips))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.clips) / self.batch_size)

    def __getitem__(self, index):
        X, y = [], []
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        for i in inds:
            video_file = os.path.join(self.video_dir, self.clips[i][0])
            frames = self.sampler(video_file, sample_id=i)
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
