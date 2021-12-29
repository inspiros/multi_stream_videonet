### Code đề tài AFOSR2022

#### Dataset AFOSR

- Dataset class: `datasets.afosr.AFOSRVideoDataset`
- Video Sampler classes: (_tất cả nằm trong file `datasets.utils.video_sampler`_)
    - `FullSampler`: lấy hết tất cả các frame - dùng để tính mean và standard deviation.
    - `SystematicSampler`: sample đều ra `n_frames` ảnh.
    - `RandomSampler`: sample ngẫu nhiên ra `n_frames` ảnh.
    - `OnceRandomSampler`: sample ngẫu nhiên ra `n_frames` nhưng các lần random sau sẽ ra giống lần đầu tiên.
    - `RandomTemporalSegmentSampler`: chia đều ra `n_frames` đoạn đều nhau và trong mỗi đoạn sample ngẫu nhiên ra 1
      frame.
    - `OnceRandomTemporalSegmentSampler`: giống `RandomTemporalSegmentSampler` nhưng các lần random sau sẽ ra giống lần
      đầu tiên (cách **Minh** đang sử dụng).
    - `LambdaSampler`: tự quy định cách lấy mẫu - advanced.

#### Code mẫu:
```python
import albumentations as A

from datasets.afosr import AFOSRVideoDataset
from datasets.utils.video_sampler import *

transform = A.Compose([
    A.Resize(128, 171, always_apply=True),
    A.CenterCrop(112, 112, always_apply=True),
    A.ToFloat(),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True),
])

train_set = AFOSRVideoDataset(
    video_dir='/mnt/disk3/datasets/afosr2022/data',
    annotation_file_path='/mnt/disk3/datasets/afosr2022/train.txt',
    sampler=OnceRandomTemporalSegmentSampler(n_frames=16),
    to_rgb=True,
    transform=transform,
    use_albumentations=True,
)
print(f'Number of train instances: {len(train_set)}')
```

Ngoài ra xem và chạy file `check_afosr_dataset.py`.
