import argparse

import cv2
import numpy as np

from datasets.ipn import *
from datasets.utils.video_sampler import *


def load_class_names(class_index_file, with_null_class=False):
    with open(class_index_file) as f:
        classes = [_.strip().split(',')[-1] for _ in f.readlines()][1:]
    return classes[1:] if not with_null_class else classes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', default='/mnt/disk3/datasets/IPN/frames/frames')
    parser.add_argument('--train_annotation_file', default='/mnt/disk3/datasets/IPN/annotations/Annot_TrainList.csv')
    parser.add_argument('--test_annotation_file', default='/mnt/disk3/datasets/IPN/annotations/Annot_TestList.csv')
    parser.add_argument('--class_index_file', default='/mnt/disk3/datasets/IPN/annotations/classIdx.csv')

    parser.add_argument('--videos_per_class', type=int, default=3)

    args = parser.parse_args()
    print(args)
    print()
    return args


def main():
    args = parse_args()
    class_names = load_class_names(args.class_index_file)

    train_set = IPNFramesDataset(
        frames_dir=args.frames_dir,
        annotation_file_path=args.train_annotation_file,
        sampler=SystematicSampler(n_frames=16),
        to_rgb=False,
        use_albumentations=True,
    )
    test_set = IPNFramesDataset(
        frames_dir=args.frames_dir,
        annotation_file_path=args.test_annotation_file,
        sampler=SystematicSampler(n_frames=16),
        to_rgb=False,
        use_albumentations=True,
    )
    print(f'[Preparing dataset] n_train_instances={len(train_set)}, n_test_instances={len(test_set)}')
    window_name = 'IPN'

    cv2.namedWindow(window_name)
    for cls in range(len(class_names)):
        count = 0
        for sample_id, (X, y) in enumerate(train_set):
            if y != cls: continue
            count += 1
            print(f'Train video {sample_id}, label={y} ({class_names[y]})')
            for img in X.permute(1, 2, 3, 0).numpy().astype(np.uint8):
                cv2.imshow(window_name, img)
                cv2.waitKey(100)
            cv2.waitKey(100)
            if count == args.videos_per_class:
                break
    print()

    for cls in range(len(class_names)):
        count = 0
        for sample_id, (X, y) in enumerate(train_set):
            if y != cls: continue
            count += 1
            print(f'Test video {sample_id}, label={y} ({class_names[y]})')
            for img in X.permute(1, 2, 3, 0).numpy().astype(np.uint8):
                cv2.imshow(window_name, img)
                cv2.waitKey(100)
            cv2.waitKey(100)
            if count == args.videos_per_class:
                break
    print()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
