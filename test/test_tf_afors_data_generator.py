from datasets.tf.afors import AFORSVideoDataGenerator
from datasets.utils.video_sampler import *


def main():
    train_generator = AFORSVideoDataGenerator(
        video_dir='/mnt/disk3/datasets/afors2022/data',
        annotation_file_path='/mnt/disk3/datasets/afors2022/train.txt',
        sampler=RandomTemporalSegmentSampler(n_frames=16),
        to_rgb=True,
        batch_size=1,
        shuffle=True,
    )
    test_generator = AFORSVideoDataGenerator(
        video_dir='/mnt/disk3/datasets/afors2022/data',
        annotation_file_path='/mnt/disk3/datasets/afors2022/test.txt',
        sampler=SystematicSampler(n_frames=16),
        to_rgb=True,
        batch_size=1,
        shuffle=True,
    )

    for X, y in train_generator:
        print(X.shape)
        print(y.shape)
