import os
import cv2
import copy
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import UserList, OrderedDict
from sortedcontainers import SortedSet, SortedList


__all__ = ['BaseDataset', 'ImageDataset', 'VideoDataset', 'SampleList']
cache_dir = 'cache'


# --------------------------------------------------
# Base datasets definitions
# --------------------------------------------------
class BaseDataset(Dataset):

    def __init__(self, name=None,
                 track_keys=None,
                 sort_keys=None,
                 transform=None):
        self.name = name if name is not None else self.__class__.__name__
        self.cache_dir = os.path.join(cache_dir, self.name)
        self._samples_init_args = dict(track_keys=track_keys, sort_keys=sort_keys)
        self.samples = SampleList(**self._samples_init_args)
        self.transform = transform

    def with_transform(self, transform):
        self.transform = transform
        return self

    @property
    def meta(self):
        return DatasetMeta(self)

    def clone(self):
        return copy.deepcopy(self)

    def empty_clone(self):
        tmp = self.samples
        self.samples = SampleList(**self._samples_init_args)
        ret = copy.deepcopy(self)
        self.samples = tmp
        return ret

    def where(self, **kwargs):
        clone = self.empty_clone()
        clone.samples = self.samples.where(**kwargs)
        return clone

    def whereid(self, **kwargs):
        clone = self.empty_clone()
        clone.samples = self.samples.whereid(**kwargs)
        return clone

    def wherenot(self, **kwargs):
        clone = self.empty_clone()
        clone.samples = self.samples.wherenot(**kwargs)
        return clone

    def whereidnot(self, **kwargs):
        clone = self.empty_clone()
        clone.samples = self.samples.whereidnot(**kwargs)
        return clone

    def categorical_map(self, key):
        return self.samples.categorical_dict(key)

    def loader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)

    def size(self):
        return len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        return self.__load__(sample)

    @staticmethod
    def __load__(sample):
        raise NotImplementedError

    def __repr__(self):
        rep = self.__class__.__name__ + '('
        rep += f'name={self.name}'
        rep += f', size={len(self)}'
        rep += f', meta={repr(self.meta)})'
        return rep


class ImageDataset(BaseDataset):

    def __init__(self, name=None,
                 track_keys=None,
                 sort_keys=None,
                 transform=None):
        super(ImageDataset, self).__init__(name, track_keys, sort_keys, transform)
        self._mean = self._std = None

    def empty_clone(self):
        tmp = (self.samples, self._mean, self._std)
        self.samples = SampleList()
        self._mean = self._std = None
        ret = copy.deepcopy(self)
        self.samples, self._mean, self._std = tmp
        return ret

    @property
    def mean(self):
        if self._mean is not None:
            return self._mean
        self._mean = 0
        for X, _ in self:
            X = X.float()
            self._mean += torch.tensor([X[_].mean() for _ in range(X.size(0))])
        self._mean /= 255. * len(self)
        return self._mean

    @property
    def std(self):
        if self._std is not None:
            return self._std
        self._std = 0
        for X, _ in self:
            X = X.float()
            self._std += torch.tensor([X[_].std() for _ in range(X.size(0))])
        self._std /= 255. * len(self)
        return self._std

    @staticmethod
    def imread(imgs, flags=None, permute=True):
        if hasattr(imgs, '__iter__') and not isinstance(imgs, str):
            ret = torch.cat([torch.from_numpy(cv2.imread(img, flags=flags)).unsqueeze(0) for img in imgs], dim=0)
        else:
            ret = torch.from_numpy(cv2.imread(imgs, flags=flags)).unsqueeze(0)
        return ret.permute((2, 0, 1)) if permute else ret

    @staticmethod
    def imwrite(tensor, file, params=None, name='hwc'):
        assert tensor.ndim == 3
        tensor = tensor.permute(name.index('h'), name.index('w'), name.index('c'))
        cv2.imwrite(file, tensor, params)

    @staticmethod
    def video_capture(video, permute=True):
        cap = cv2.VideoCapture(video)
        frames = []
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_id in range(frames_count):
            _, frame = cap.read()
            frames.append(torch.from_numpy(frame).unsqueeze(0))
        cap.release()
        ret = torch.cat(frames, dim=0)
        return ret.permute((0, 3, 1, 2)) if permute else ret

    @staticmethod
    def video_write(tensor, file, fps=15, name='nhwc'):
        assert tensor.ndim == 4
        tensor = tensor.permute(name.index('n'), name.index('h'),
                                name.index('w'), name.index('c')).numpy().astype(np.uint8)
        shapes = tensor.shape
        out = cv2.VideoWriter(file,
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              fps,
                              (shapes[2], shapes[1]))
        for _ in range(shapes[0]):
            out.write(tensor[_])
        out.release()

    @staticmethod
    def bgr2rgb(img):
        return img[..., [2, 1, 0]]

    @staticmethod
    def rgb2bgr(img):
        return img[..., [2, 1, 0]]


class VideoDataset(ImageDataset):

    def __init__(self, name=None,
                 track_keys=None,
                 sort_keys=None,
                 transform=None,
                 spatial_transform=None):
        super(VideoDataset, self).__init__(name, track_keys, sort_keys, transform)
        self.spatial_transform = None
        self.with_spatial_transform(spatial_transform)

    def with_spatial_transform(self, spatial_transform):
        self.spatial_transform = spatial_transform
        return self

    @property
    def mean(self):
        if self._mean is not None:
            return self._mean
        self._mean = 0
        for X, _ in self:
            X = X.float()
            self._mean += torch.tensor([X[:, _].mean() for _ in range(X.size(1))])
        self._mean /= 255. * len(self)
        return self._mean

    @property
    def std(self):
        if self._std is not None:
            return self._std
        self._std = 0
        for X, _ in self:
            X = X.float()
            self._std += torch.tensor([X[:, _].std() for _ in range(X.size(1))])
        self._std /= 255. * len(self)
        return self._std


# --------------------------------------------------
# Sample holders for datasets
# --------------------------------------------------
class SampleList(UserList):

    def __init__(self, initlist=None, track_keys=None, sort_keys=None):
        self.sort_keys = set() if sort_keys is None else sort_keys
        self.track_keys = set() if track_keys is None else track_keys
        self.keys = set()

        if hasattr(self.sort_keys, '__len__') and len(self.sort_keys) > 0:
            super(SampleList, self).__init__()
            # override elements data with a sorted list to keep thing sorted
            self.data: SortedList = SortedList(initlist, key=self.__keyfunc__)
            self._append = self.data.add
        else:
            super(SampleList, self).__init__(initlist)
            self._append = self.data.append
        if len(self.data) > 0:
            self.keys.update(self.data[0].keys())
        self.uniques = {key: SortedSet() for key in self.keys if key in self.track_keys}
        self.counts = {key: OrderedDict() for key in self.keys if key in self.track_keys}
        for item in self.data:
            for key in self.track_keys:
                if key in self.keys:
                    self.uniques[key].add(item[key])
                    if item[key] not in self.counts[key]:
                        self.counts[key][item[key]] = 1
                    else:
                        self.counts[key][item[key]] += 1

    def __keyfunc__(self, sample):
        return tuple(sample[key] for key in self.sort_keys)

    def track(self, *track_keys):
        for track_key in track_keys:
            if isinstance(track_key, str):
                self.track_keys.add(track_key)
            elif hasattr(track_key, '__iter__'):
                self.track_keys.update(track_key)
            else:
                self.track_keys.add(str(track_key))

    def sort_by(self, *sort_keys):
        if len(self.sort_keys) > 0:
            raise ValueError('sorted_keys must be declared in class __init__')
        else:
            sort_keys = tuple(key for key in sort_keys if key in self.keys)
            getattr(self.data, 'sort')(key=lambda s: tuple(s[sort_key] for sort_key in sort_keys))

    def append(self, item):
        if len(self) == 0:
            self.keys.update(item.keys())
            self.uniques = {key: SortedSet() for key in self.keys if key in self.track_keys}
            self.counts = {key: OrderedDict() for key in self.keys if key in self.track_keys}
        for key in self.track_keys:
            if key in self.keys:
                self.uniques[key].add(item[key])
                if item[key] not in self.counts[key]:
                    self.counts[key][item[key]] = 1
                else:
                    self.counts[key][item[key]] += 1
        self._append(item)

    def update(self, items):
        for item in items:
            self.append(item)

    def where(self, **kwargs):
        for key in set(kwargs.keys()).difference(self.keys):
            del kwargs[key]
        for key, val in kwargs.items():
            if hasattr(val, '__iter__') and not isinstance(val, str):
                kwargs[key] = [str(v) for v in val]
            else:
                kwargs[key] = str(val)
        return SampleList(
            initlist=filter(lambda sample: self.__isvalid__(sample, kwargs), self.data),
            track_keys=self.track_keys,
            sort_keys=self.sort_keys)

    def whereid(self, **kwargs):
        for key in set(kwargs.keys()).difference(self.keys):
            del kwargs[key]
        for key, val in kwargs.items():
            if hasattr(val, '__iter__') and not isinstance(val, str):
                kwargs[key] = [self.uniques[key][v] if v >= 0 else self.uniques[key][::-1][-v-1] for v in val]
            else:
                kwargs[key] = self.uniques[key][val]
        return self.where(**kwargs)

    def wherenot(self, **kwargs):
        for key in set(kwargs.keys()).difference(self.keys):
            del kwargs[key]
        for key, val in kwargs.items():
            if hasattr(val, '__iter__') and not isinstance(val, str):
                kwargs[key] = [str(v) for v in val]
            else:
                kwargs[key] = str(val)
        return SampleList(
            initlist=filter(lambda sample: self.__isinvalid__(sample, kwargs), self.data),
            track_keys=self.track_keys,
            sort_keys=self.sort_keys)

    def whereidnot(self, **kwargs):
        for key in set(kwargs.keys()).difference(self.keys):
            del kwargs[key]
        for key, val in kwargs.items():
            if hasattr(val, '__iter__') and not isinstance(val, str):
                kwargs[key] = [self.uniques[key][v] if v >= 0 else self.uniques[key][::-1][-v-1] for v in val]
            else:
                kwargs[key] = self.uniques[key][val]
        return self.wherenot(**kwargs)

    @staticmethod
    def __isvalid__(sample, predicates):
        if len(predicates) == 0:
            return False
        return all(str(sample[key]) == val or
                   (hasattr(val, '__iter__') and str(sample[key]) in val)
                   for key, val in predicates.items())

    @staticmethod
    def __isinvalid__(sample, predicates):
        return not any(str(sample[key]) == val or
                       (hasattr(val, '__iter__') and str(sample[key]) in val)
                       for key, val in predicates.items())

    def categorical(self, sample, key):
        return self.uniques[key].index(sample[key])

    def categorical_dict(self, key):
        return {uval: self.uniques[key].index(uval)
                for uval in self.uniques[key]}

    def onehot(self, sample, key):
        vec = [0 for _ in range(len(self.uniques[key]))]
        vec[self.categorical(sample, key)] = 1
        return vec

    def onehot_dict(self, key):
        return {uval: [1 if uval == self.uniques[key] else 0 for _ in range(len(self.uniques[key]))]
                for uval in self.uniques[key]}


class DatasetMeta:

    def __init__(self, dataset: BaseDataset):
        self.sample_list = dataset.samples

    @property
    def keys(self):
        return self.sample_list.keys

    @property
    def track_keys(self):
        return self.sample_list.track_keys

    @property
    def sort_keys(self):
        return self.sample_list.sort_keys

    @property
    def uniques(self):
        return self.sample_list.uniques

    @property
    def counts(self):
        return self.sample_list.counts

    def __repr__(self):
        rep = self.__class__.__name__ + '('
        rep += f'keys={self.keys}'
        rep += f', uniques={self.uniques})'
        return rep
