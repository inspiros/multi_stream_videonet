import argparse
import os

import albumentations as A
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from torch.utils.data import DataLoader

from datasets.afors import *
from datasets.utils.mean_std_estimator import compute_mean_std
from datasets.utils.video_sampler import *
from models import r2plus1d_18
# from models.videos.c3d import c3d_bn
from utils.plot_utils import *
from utils.trainer_utils import ClassificationTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--train_annotation_file', required=True)
    parser.add_argument('--test_annotation_file', required=True)

    parser.add_argument('--weights', default='',
                        help='pretrained weights path.')
    parser.add_argument('--mean_std_file', default='',
                        help='path to output folder.')
    parser.add_argument('--output_dir', default='outputs/AFORS',
                        help='path to output folder.')

    parser.add_argument('--crop_size', type=int, default=112,
                        help='center crop size.')
    parser.add_argument('--temporal_slice', type=int, default=16,
                        help='temporal length of each sample.')

    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--val_frequency', type=int, default=1,
                        help='number of epochs between each validation.')
    parser.add_argument('--save_frequency', type=int, default=1,
                        help='number of epochs between each checkpoint.')

    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.train_batch_size is None:
        args.train_batch_size = args.batch_size
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    print(args)
    print()
    return args


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_file = os.path.join(args.output_dir, 'checkpoint.pt')
    model_file = os.path.join(args.output_dir, 'model.pt')
    result_file = os.path.join(args.output_dir, 'result.yaml')
    accuracy_figure_file = os.path.join(args.output_dir, 'accuracy.png')
    loss_figure_file = os.path.join(args.output_dir, 'loss.png')
    confusion_matrix_figure_file = os.path.join(args.output_dir, 'confusion_matrix.png')

    if len(args.mean_std_file):  # compute or load mean/std of training set
        if os.path.exists(args.mean_std_file):
            mean_std_dict = torch.load(args.mean_std_file, map_location='cpu')
            mean = mean_std_dict['mean']
            std = mean_std_dict['std']
            del mean_std_dict
        else:
            train_set = AFORSVideoDataset(
                video_dir=args.video_dir,
                annotation_file_path=args.train_annotation_file,
                sampler=FullSampler(),
                transform=A.ToFloat(),
                use_albumentations=True,
            )
            mean, std = compute_mean_std(train_set)
            torch.save({'mean': mean, 'std': std}, args.mean_std_file)
    else:  # ImageNet's mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transform = A.Compose([
        A.Resize(128, 171, always_apply=True),
        A.CenterCrop(112, 112, always_apply=True),
        A.ToFloat(),
        A.Normalize(mean=mean,
                    std=std,
                    always_apply=True),
    ])

    train_set = AFORSVideoDataset(
        video_dir=args.video_dir,
        annotation_file_path=args.train_annotation_file,
        sampler=RandomTemporalSegmentSampler(n_frames=args.temporal_slice),
        transform=transform,
        use_albumentations=True,
    )
    test_set = AFORSVideoDataset(
        video_dir=args.video_dir,
        annotation_file_path=args.test_annotation_file,
        sampler=SystematicSampler(n_frames=args.temporal_slice),
        transform=transform,
        use_albumentations=True,
    )
    print(f'[Preparing dataset] n_train_instances={len(train_set)}, n_test_instances={len(test_set)}')
    # not named yet
    class_names = np.unique(train_set.labels).astype(str).tolist()

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

    model = r2plus1d_18(num_classes=len(class_names),
                        pretrained=True,
                        progress=True)
    # model = c3d_bn(num_classes=len(class_names),
    #                temporal_slice=args.temporal_slice)
    model = model.to(args.device)
    # load pre-trained weights
    if len(args.weights):
        assert os.path.exists(args.weights), f"{args.weights} does not exists."
        model.load_state_dict(torch.load(args.weights, map_location=args.device))

    # set batchnorm's behavior
    def update_batchnorm_momentum(m):
        if isinstance(m, torch.nn.BatchNorm3d):
            # m.momentum = 0.01
            m.track_running_stats = False
            m.register_buffer("running_mean", None)
            m.register_buffer("running_var", None)
            m.register_buffer("num_batches_tracked", None)

    model.apply(update_batchnorm_momentum)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )

    trainer = ClassificationTrainer(model, optimizer, criterion, device=args.device, verbose=True)

    train_losses, train_accuracies = {}, {}
    test_losses, test_accuracies = {}, {}
    start_epoch = 0

    # resume from previous states
    if os.path.exists(checkpoint_file):
        state_dicts = torch.load(checkpoint_file, map_location=args.device)
        start_epoch = state_dicts['epoch']
        print(f'Continuing from epoch {start_epoch}')
        model.load_state_dict(state_dicts['model_state_dict'])
        optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
        scheduler.load_state_dict(state_dicts['scheduler_state_dict'])

        train_losses = state_dicts['train_losses']
        train_accuracies = state_dicts['train_accuracies']
        test_losses = state_dicts['test_losses']
        test_accuracies = state_dicts['test_accuracies']
        del state_dicts

    # training and validation loop
    if start_epoch < args.max_epoch - 1:
        print('\n\n[Training]')
        for epoch in range(start_epoch, args.max_epoch):
            print(f"[Epoch {epoch + 1} / {args.max_epoch}]")
            train_epoch_loss, train_epoch_accuracy = trainer.fit(train_loader)
            train_losses[epoch + 1] = train_epoch_loss
            train_accuracies[epoch + 1] = train_epoch_accuracy
            print(f'[Epoch {epoch + 1} / {args.max_epoch}] '
                  f'train_loss={train_epoch_loss:.4f}, '
                  f'train_accuracy={train_epoch_accuracy:.2f}')

            if epoch % args.val_frequency == 0 or epoch == args.epoch - 1:
                test_epoch_loss, test_epoch_accuracy = trainer.validate(test_loader)
                test_losses[epoch + 1] = test_epoch_loss
                test_accuracies[epoch + 1] = test_epoch_accuracy
                print(f'[Epoch {epoch + 1} / {args.max_epoch}] '
                      f'test_loss={test_epoch_loss:.4f}, '
                      f'test_accuracy={test_epoch_accuracy:.2f}')

            scheduler.step(train_epoch_loss)

            # save state dicts after each epoch
            if epoch % args.save_frequency == 0 or epoch == args.epochs - 1:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'test_losses': test_losses,
                    'test_accuracies': test_accuracies,
                }, checkpoint_file)
        # serialize latest weights to disk
        torch.save(model.state_dict(), model_file)

    # testing
    y_trues, y_preds = trainer.test(test_loader)
    print('\n\n[Testing]')
    cm = confusion_matrix(y_trues, y_preds)
    accuracy = accuracy_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds, average='weighted')
    recall = recall_score(y_trues, y_preds, average='weighted')
    f1 = f1_score(y_trues, y_preds, average='weighted')
    print(f'Confusion matrix:\n{cm}')
    print(f'Accuracy: {accuracy:.02%}')
    print(f'Precision: {precision:.02%}')
    print(f'Recall: {recall:.02%}')
    print(f'F1: {f1:.02%}')

    # save results
    with open(result_file, 'w') as rf:
        yaml.safe_dump({
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'confusion_matrix': cm.tolist(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }, rf, indent=4, default_flow_style=None, sort_keys=False)

    plot_metrics(train_losses, test_losses, label='Losses', save_file=loss_figure_file)
    plot_metrics(train_accuracies, test_accuracies, label='Accuracies', save_file=accuracy_figure_file)
    plot_confusion_matrix(cm, class_names=class_names, save_file=confusion_matrix_figure_file)


if __name__ == '__main__':
    main()
