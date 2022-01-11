### Code đề tài AFORS2022

#### Dataset AFORS:

- Dataset class: `datasets.afors.AFORSVideoDataset`
- Video Sampler classes: (_tất cả nằm trong file `datasets.utils.video_sampler`_)
    - `FullSampler`: lấy hết tất cả các frame - dùng để tính mean và standard deviation.
    - `SystematicSampler`: sample đều ra `n_frames` ảnh.
    - `RandomSampler`: sample ngẫu nhiên ra `n_frames` ảnh.
    - `OnceRandomSampler`: sample ngẫu nhiên ra `n_frames` nhưng các lần random sau sẽ ra giống lần đầu tiên.
    - `RandomTemporalSegmentSampler`: chia đều ra `n_frames` đoạn đều nhau và trong mỗi đoạn sample ngẫu nhiên ra 1
      frame.
    - `OnceRandomTemporalSegmentSampler`: giống `RandomTemporalSegmentSampler` nhưng các lần random sau sẽ ra giống lần
      đầu tiên.
    - `LambdaSampler`: tự quy định cách lấy mẫu - advanced.

Thống nhất sử dụng `RandomTempỏalSegmentSampler` cho tập train và `SystematicSampler` cho tập val/test như code mẫu sau:

```python
import albumentations as A

from datasets.afors import AFORSVideoDataset
from datasets.utils.video_sampler import *

transform = A.Compose([
    A.Resize(128, 171, always_apply=True),
    A.CenterCrop(112, 112, always_apply=True),
    A.ToFloat(),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True),
])

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
    annotation_file_path='/mnt/disk3/datasets/afors2022/test.txt',
    sampler=SystematicSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
)
print(f'Number of train instances: {len(train_set)}')
print(f'Number of test instances: {len(test_set)}')
```

Ngoài ra xem và chạy file `check_afors_dataset.py`.

#### Model:

Một số model Conv3D có sẵn trong package `models.videos`.

#### Note:

_Anh Hiếu or anh Quân gánh xem nào_
