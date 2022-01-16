import albumentations as A

from datasets.tf.ipn import IPNFramesDataGenerator
from datasets.utils.video_sampler import *


def main():
    # configurations
    frames_dir = '/mnt/disk3/datasets/IPN/frames/frames'
    train_annotation_file = '/mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv'
    test_annotation_file = '/mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv'
    temporal_slice = 16  # number of sampled frames
    batch_size = 8
    data_format = 'channels_last'  # channels_first or [channels_last]

    # image transform
    transform = A.Compose([
        A.Resize(128, 171, always_apply=True),
        A.CenterCrop(112, 112, always_apply=True),
        A.ToFloat(),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    always_apply=True),
    ])

    train_generator = IPNFramesDataGenerator(
        frames_dir=frames_dir,
        annotation_file_path=train_annotation_file,
        sampler=RandomTemporalSegmentSampler(n_frames=temporal_slice),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
        data_format=data_format,
        batch_size=batch_size,
        shuffle=True,
    )
    test_generator = IPNFramesDataGenerator(
        frames_dir=frames_dir,
        annotation_file_path=test_annotation_file,
        sampler=SystematicSampler(n_frames=temporal_slice),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
        data_format=data_format,
        batch_size=batch_size,
        shuffle=False,
    )

    # properties
    print('\nDataset loaded:')
    print(f'Number of classes: {train_generator.n_classes}')
    print(f'Number of training instances: {len(train_generator.clips)}, '
          f'number of training batches: {len(train_generator)}')
    print(f'Number of testing instances: {len(test_generator.clips)}, '
          f'number of testing batches: {len(test_generator)}')

    # sample loop
    print('\nSample loop:')
    for batch_id, (X, y) in enumerate(train_generator):
        print(f'[{batch_id + 1}/{len(train_generator)}] X: {X.shape}, y: {y}')
        if batch_id == 3:
            break


if __name__ == '__main__':
    main()
