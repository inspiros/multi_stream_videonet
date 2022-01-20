import albumentations as A
from torch.utils.data import DataLoader

from datasets.ipn import IPNFramesDataset
from datasets.utils.video_sampler import *


def main():
    # configurations
    frames_dir = '/mnt/disk3/datasets/IPN/frames/frames'
    train_annotation_file = '/mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv'
    test_annotation_file = '/mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv'
    batch_size = 8

    # image transform
    transform = A.Compose([
        A.CenterCrop(224, 224, always_apply=True),
        A.ToFloat(),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    always_apply=True),
    ])

    train_set = IPNFramesDataset(
        frames_dir=frames_dir,
        annotation_file_path=train_annotation_file,
        sampler=RandomTemporalSegmentSampler(n_frames=16),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
    )
    test_set = IPNFramesDataset(
        frames_dir=frames_dir,
        annotation_file_path=test_annotation_file,
        sampler=SystematicSampler(n_frames=16),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # properties
    print('\nDataset loaded:')
    print(f'Number of classes: {train_set.n_classes}')
    print(f'Number of training instances: {len(train_set)}, '
          f'number of training batches: {len(train_loader)}')
    print(f'Number of testing instances: {len(test_set)}, '
          f'number of testing batches: {len(test_loader)}')

    # sample loop
    print('\nSample loop:')
    for batch_id, (X, y) in enumerate(train_loader):
        print(f'[{batch_id + 1}/{len(train_loader)}] X: {X.shape}, y: {y}')
        if batch_id == 3:
            break


if __name__ == '__main__':
    main()
