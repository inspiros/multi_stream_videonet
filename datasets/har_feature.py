import os
import sys
from collections import namedtuple

import torch
from scipy.io import loadmat
from tqdm import tqdm

from ._base import *

__all__ = ['har_feature', 'loso_har_feature']

ReturnType = namedtuple('HoldoutReturnType', ['X_train', 'y_train', 'X_test', 'y_test'])


class loso_har_feature(BaseDataset):

    def __init__(self,
                 root=None,
                 views=None,
                 map_location=None):
        super().__init__(track_keys=('train_view', 'test_actor'),
                         sort_keys=('train_view', 'test_actor'))
        self.root = root
        self.training = True
        self.map_location = map_location

        annotations = []
        for train_view in os.listdir(self.root) if views is None else views:
            train_view_dir = os.path.join(self.root, train_view)
            for test_actor in os.listdir(train_view_dir):
                test_actor_dir = os.path.join(train_view_dir, test_actor)
                train_mat = os.path.join(test_actor_dir, 'train.mat')
                test_mat = os.path.join(test_actor_dir, 'test.mat')
                annotations.append((train_view, test_actor, train_mat, test_mat))
        for annotation in tqdm(annotations, desc=self.name, file=sys.stdout):
            sample = dict(train_view=annotation[0],
                          test_actor=annotation[1],
                          train_mat=annotation[2],
                          test_mat=annotation[3],
                          )
            self.samples.append(sample)

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def __getitem__(self, item):
        if len(self.meta.uniques['test_actor']) > 1:
            raise ValueError(f"Only support leave one subject out cross-validation procedure, "
                             f"more than 1 test actor detected: {self.meta.uniques['test_actor']}")
        sample = self.samples[item]
        if isinstance(sample, SampleList) or isinstance(sample, list):
            rets = [self.__load__(s) for s in sample]
            return ReturnType([ret[0] for ret in rets], rets[0][1], [ret[2] for ret in rets], rets[0][3])
        return self.__load__(sample)

    def __load__(self, sample):
        train_mat = loadmat(sample['train_mat'])
        test_mat = loadmat(sample['test_mat'])
        return ReturnType(torch.from_numpy(train_mat['data']).float().to(self.map_location),
                          torch.from_numpy(train_mat['label']).long().squeeze().to(self.map_location),
                          torch.from_numpy(test_mat['data']).float().to(self.map_location),
                          torch.from_numpy(test_mat['label']).long().squeeze().to(self.map_location))

    def __repr__(self):
        rep = self.__class__.__name__ + '('
        rep += f'name={self.name}'
        rep += f', root={self.root.replace(os.sep, "/")}'
        rep += f', size={len(self)}'
        rep += f', meta={repr(self.meta)})'
        return rep


class har_feature(BaseDataset):

    def __init__(self,
                 root=None,
                 views=None,
                 map_location=None):
        super().__init__(track_keys=('train_view',),
                         sort_keys=('train_view',))
        self.root = root
        self.training = True
        self.map_location = map_location

        annotations = []
        for train_view in os.listdir(self.root) if views is None else views:
            train_view_dir = os.path.join(self.root, train_view)
            train_mat = os.path.join(train_view_dir, 'train.mat')
            test_mat = os.path.join(train_view_dir, 'test.mat')
            if not os.path.isfile(train_mat) or not os.path.isfile(test_mat):
                print(f'Skipping "{train_view_dir}" as train and test mat cannot be found.')
                continue
            annotations.append((train_view, train_mat, test_mat))
        for annotation in tqdm(annotations, desc=self.name, file=sys.stdout):
            sample = dict(train_view=annotation[0],
                          train_mat=annotation[1],
                          test_mat=annotation[2],
                          )
            self.samples.append(sample)

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def __getitem__(self, item):
        sample = self.samples[item]
        if isinstance(sample, SampleList) or isinstance(sample, list):
            rets = [self.__load__(s) for s in sample]
            return ReturnType([ret[0] for ret in rets], rets[0][1], [ret[2] for ret in rets], rets[0][3])
        return self.__load__(sample)

    def __load__(self, sample):
        train_mat = loadmat(sample['train_mat'])
        test_mat = loadmat(sample['test_mat'])
        return ReturnType(torch.from_numpy(train_mat['data']).float().to(self.map_location),
                          torch.from_numpy(train_mat['label']).long().squeeze().to(self.map_location),
                          torch.from_numpy(test_mat['data']).float().to(self.map_location),
                          torch.from_numpy(test_mat['label']).long().squeeze().to(self.map_location))

    def __repr__(self):
        rep = self.__class__.__name__ + '('
        rep += f'name={self.name}'
        rep += f', root={self.root.replace(os.sep, "/")}'
        rep += f', size={len(self)}'
        rep += f', meta={repr(self.meta)})'
        return rep
