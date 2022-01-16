# Data loader classes for AFORS2022 research project
This branch contains data loader classes implemented for `torch` and `tensorflow` frameworks.

### Requirements:
- `numpy`
- `cv2`
- `albumentations`
- `torch` or `tensorflow`

### Instruction:
#### 0. Video samplers
Some video samplers are implemented in `datasets.utils.video_sampler` package.
These are used in initialization of data generator classes (see below).
Convention:
- Use `datasets.utils.video_sampler.RandomTemporalSegmentSampler` for training set.
- Use `datasets.utils.video_sampler.SystematicSampler` for testing set.
Please see the examples below.

#### 1. IPN RGB/Flow dataset
##### 1.1. Torch Dataset `datasets.afors.IPNFramesDataset`
Initialization:

```python
import albumentations as A

from torch.utils.data import DataLoader
from datasets.ipn import IPNFramesDataset
from datasets.utils.video_sampler import *


train_set = IPNFramesDataset(
    frames_dir='/mnt/disk3/datasets/IPN/frames/frames',
    annotation_file_path='/mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv',
    sampler=RandomTemporalSegmentSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
)
test_set = IPNFramesDataset(
    frames_dir='/mnt/disk3/datasets/IPN/frames/frames',
    annotation_file_path='/mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv',
    sampler=SystematicSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
```

##### 1.2. Tensorflow Data Generator `datasets.tf.ipn.IPNFramesDataGenerator`
Initialization:

```python
import albumentations as A

from datasets.tf.ipn import IPNFramesDataGenerator
from datasets.utils.video_sampler import *


train_generator = IPNFramesDataGenerator(
    video_dir='/mnt/disk3/datasets/IPN/frames/frames',
    annotation_file_path='/mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv',
    sampler=RandomTemporalSegmentSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
    data_format='channels_last',
    batch_size=16,
    shuffle=True,
)
test_generator = IPNFramesDataGenerator(
    video_dir='/mnt/disk3/datasets/IPN/frames/frames',
    annotation_file_path='/mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv',
    sampler=SystematicSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
    data_format='channels_last',
    batch_size=16,
    shuffle=False,
)
```

#### 2. AFORS RGB/Flow dataset
##### 2.1. Torch Dataset `datasets.afors.AFORSVideoDataset`
Initialization:

```python
import albumentations as A

from torch.utils.data import DataLoader
from datasets.afors import AFORSVideoDataset
from datasets.utils.video_sampler import *


train_set = AFORSVideoDataset(
    video_dir='/mnt/disk3/datasets/afors2022/data',
    annotation_file_path='/mnt/disk3/datasets/afors2022/train.txt',
    sampler=RandomTemporalSegmentSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
)
test_set = AFORSVideoDataset(
    video_dir='/mnt/disk3/datasets/afors2022/data',
    annotation_file_path='/mnt/disk3/datasets/afors2022/val.txt',
    sampler=SystematicSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
```

##### 2.2. Tensorflow Data Generator `datasets.tf.afors.AFORSVideoDataGenerator`
Initialization:

```python
import albumentations as A

from datasets.tf.afors import AFORSVideoDataGenerator
from datasets.utils.video_sampler import *


train_generator = AFORSVideoDataGenerator(
    video_dir='/mnt/disk3/datasets/afors2022/data',
    annotation_file_path='/mnt/disk3/datasets/afors2022/train.txt',
    sampler=RandomTemporalSegmentSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
    data_format='channels_last',
    batch_size=16,
    shuffle=True,
)
test_generator = AFORSVideoDataGenerator(
    video_dir='/mnt/disk3/datasets/afors2022/data',
    annotation_file_path='/mnt/disk3/datasets/afors2022/val.txt',
    sampler=SystematicSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
    data_format='channels_last',
    batch_size=16,
    shuffle=False,
)
```

### Note:

- Have a look at the test python files.
- Most `keras`'s layers only support `data_format='channels_last'`.
- For `tensorflow`'s data generators, pass them as first argument to the `model.fit` and `model.evaluate` methods:
    ```python
    # compile a keras model then fit/evaluate with the above generators
    model.fit(x=train_generator, steps_per_epoch=len(train_generator), epochs=30)
    model.evaluate(x=test_generator, steps=len(test_generator))
    ```
