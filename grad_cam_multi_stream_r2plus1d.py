import argparse
import os
from collections.abc import Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import Dataset

from models import multi_stream_r2plus1d_18

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f')
ap.add_argument('--model_file', default='outputs/RGB_OF/fold_1/model.pt',
                help='path to save the trained model.')
ap.add_argument('--rgb_data_dir', default='D:/datasets/AFORS2022/truongvanminh/data_RGB',
                help='path to RGB data folder.')
ap.add_argument('--of_data_dir', default='D:/datasets/AFORS2022/truongvanminh/data_OF',
                help='path to OF data folder.')
ap.add_argument('--rgb_data_file', default='D:/datasets/AFORS2022/truongvanminh/data_RGB.csv',
                help='path to RGB data file.')
ap.add_argument('--of_data_file', default='D:/datasets/AFORS2022/truongvanminh/data_OF.csv',
                help='path to OF data file.')
ap.add_argument('--rgb_mean_std', default='D:/datasets/AFORS2022/truongvanminh/data_RGB.pt',
                help='path to RGB mean and std file.')
ap.add_argument('--of_mean_std', default='D:/datasets/AFORS2022/truongvanminh/data_OF.pt',
                help='path to OF mean and std file.')
ap.add_argument('--label_file', default='D:/datasets/AFORS2022/truongvanminh/label.txt',
                help='path to label file.')
ap.add_argument('--output_dir', default='outputs/RGB_OF',
                help='path to output folder.')

ap.add_argument('--seed', type=int, default=42,
                help='seed for random generator.')

ap.add_argument('--batch_size', type=int, default=1)
args = ap.parse_args()

device = torch.device('cpu')

with open(args.label_file, 'r') as f:
    class_names = [_.strip() for _ in f.readlines()]

# load data
rgb_df = pd.read_csv(args.rgb_data_file)
of_df = pd.read_csv(args.of_data_file)
X_RGB = np.array([os.path.join(args.rgb_data_dir, _) for _ in rgb_df['clip']])
X_OF = np.array([os.path.join(args.of_data_dir, _) for _ in of_df['clip']])
y = rgb_df['label'].to_numpy()
del rgb_df, of_df

print('Xs:', len(X_RGB), len(X_OF))
print('y:', len(y))


# custom dataset
class MultiStreamUCF11(Dataset):
    def __init__(self, multi_stream_clips, targets, transform=None, dtype=torch.float32, device='cpu'):
        assert len(multi_stream_clips) > 0
        self.num_streams = len(multi_stream_clips)
        self.multi_stream_clips = multi_stream_clips
        self.targets = targets
        self.labels = np.unique(self.targets).tolist()
        self.target2id = lambda _: self.labels.index(_)
        self.y = torch.tensor([self.target2id(_) for _ in self.targets],
                              dtype=torch.long)
        self.transform = transform
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return len(self.multi_stream_clips[0])

    def __getitem__(self, i):
        multi_stream_input_frames = []
        for stream_id in range(self.num_streams):
            clip = self.multi_stream_clips[stream_id][i]
            cap = cv2.VideoCapture(clip)
            assert cap.isOpened()
            input_frames = []
            for frame_id in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                # image = Image.open(frame)
                # image = image.convert('RGB')
                # image = np.array(image)
                _, image = cap.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.transform is not None:
                    if isinstance(self.transform, Sequence):
                        image = self.transform[stream_id](image=image)['image']
                    else:
                        image = self.transform(image=image)['image']
                input_frames.append(image)
            cap.release()

            # fill optical flow missing frame
            while len(input_frames) < 16:
                input_frames.insert(0, np.zeros_like(input_frames[-1]))
            input_frames = np.asarray(input_frames)
            input_frames = np.transpose(input_frames, (3, 0, 1, 2))
            input_frames = torch.from_numpy(input_frames).to(dtype=self.dtype,
                                                             device=self.device)
            multi_stream_input_frames.append(input_frames)
        # label
        y = self.y[i].to(self.device)
        return *multi_stream_input_frames, y


# transform = A.Compose([
#     A.Resize(128, 171, always_apply=True),
#     A.CenterCrop(112, 112, always_apply=True),
#     A.Normalize(mean=[0.43216, 0.394666, 0.37645],
#                 std=[0.22803, 0.22145, 0.216989],
#                 always_apply=True)
# ])
rgb_mean_std = torch.load(args.rgb_mean_std)
of_mean_std = torch.load(args.of_mean_std)
rgb_transform = A.Compose([
    A.Resize(128, 171, always_apply=True),
    A.CenterCrop(112, 112, always_apply=True),
    A.Normalize(mean=rgb_mean_std['mean'] / 255,
                std=rgb_mean_std['std'] / 255,
                always_apply=True)
])
of_transform = A.Compose([
    A.Resize(128, 171, always_apply=True),
    A.CenterCrop(112, 112, always_apply=True),
    A.Normalize(mean=of_mean_std['mean'] / 255,
                std=of_mean_std['std'] / 255,
                always_apply=True)
])
del rgb_mean_std, of_mean_std

dataset = MultiStreamUCF11([X_RGB, X_OF], y,
                           transform=[rgb_transform, of_transform],
                           device=device)

# model
model = multi_stream_r2plus1d_18(num_classes=len(class_names),
                                 num_streams=2,
                                 fusion_stage=4,
                                 transfer_stages=(4,),
                                 pretrained=True,
                                 progress=True)
model = model.to(device)
# load pre-trained model
if len(args.model_file) and os.path.exists(args.model_file):
    model.load_state_dict(torch.load(args.model_file, map_location=device))

cam_rgb = GradCAM(model=model,
                  target_layers=[model.layer3[0][-1]],
                  use_cuda=device.type == 'cuda')
cam_of = GradCAM(model=model,
                 target_layers=[model.layer3[1][-1]],
                 use_cuda=device.type == 'cuda')
target_size = (512, 512)
cam_rgb.target_size = target_size
cam_of.target_size = target_size

while True:
    sample_id = np.random.randint(0, len(dataset))

    clips = [dataset.multi_stream_clips[stream_id][sample_id] for stream_id in range(dataset.num_streams)]
    X_RGB, X_OF, y = dataset[sample_id]
    print(f'Sample {sample_id} - Clips: {clips} - Label: {y.item()}')

    X_RGB = X_RGB.unsqueeze(0)
    X_OF = X_OF.unsqueeze(0)
    y = y.unsqueeze(0)

    grayscale_cam_rgb = cam_rgb(input_tensor=[X_RGB, X_OF], target_category=y)[0]
    grayscale_cam_of = cam_of(input_tensor=[X_RGB, X_OF], target_category=y)[0]

    rgb_cap = cv2.VideoCapture(clips[0])
    of_cap = cv2.VideoCapture(clips[1])
    for frame_id in range(16):
        _, rgb_img = rgb_cap.read()
        rgb_img = cv2.resize(rgb_img, target_size)
        rgb_img = rgb_img.astype(np.float32) / 255.

        if frame_id == 0:
            of_img = np.zeros_like(rgb_img)
        else:
            _, of_img = of_cap.read()
        of_img = cv2.resize(of_img, target_size)
        of_img = of_img.astype(np.float32) / 255.

        grayscale_cam_rgb_id = round(frame_id / 15 * (grayscale_cam_rgb.shape[0] - 1))
        grayscale_cam_of_id = round(frame_id / 15 * (grayscale_cam_of.shape[0] - 1))

        rgb_cam_vis = show_cam_on_image(rgb_img, grayscale_cam_rgb[grayscale_cam_rgb_id], use_rgb=False)
        of_cam_vis = show_cam_on_image(of_img, grayscale_cam_of[grayscale_cam_of_id], use_rgb=False)
        cv2.imshow('RGB-OF GradCam', np.concatenate([rgb_cam_vis, of_cam_vis], axis=1))
        cv2.waitKey(200)

    rgb_cap.release()
    of_cap.release()
    key = cv2.waitKey()
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
