import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.video_sampler import *

__all__ = ['IPNVideoDataset', 'IPNFramesDataset']


class IPNVideoDataset(Dataset):
    def __init__(self,
                 video_dir,
                 annotation_file_path,
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 ):
        self.video_dir = video_dir
        self.annotation_file_path = annotation_file_path
        self.sampler = sampler
        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

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

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        video_file, start_frame, end_frame = self.clips[item]
        video_file = os.path.join(self.video_dir, video_file)
        frames = self.sampler(video_file, start_frame, end_frame, sample_id=item)
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]
        data = torch.from_numpy(np.stack(frames).transpose((3, 0, 1, 2)))
        return data, self.labels[item]


class IPNFramesDataset(Dataset):
    def __init__(self,
                 frames_dir,
                 annotation_file_path,
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 ):
        self.frames_dir = frames_dir
        self.annotation_file_path = annotation_file_path
        self.sampler = sampler
        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

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

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        video, start_frame, end_frame = self.clips[item]
        frames = self.sampler([os.path.join(self.frames_dir, video, f'{video}_{frame_id + 1:06d}.jpg')
                               for frame_id in range(start_frame, end_frame + 1)], sample_id=item)
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]
        data = torch.from_numpy(np.stack(frames).transpose((3, 0, 1, 2)))
        return data, self.labels[item]
