import argparse
import os
import time
from collections.abc import Sequence

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import multi_stream_r2plus1d_18

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f')
ap.add_argument('--model_file', default='',
                help='path to save the trained model.')
ap.add_argument('--rgb_data_dir', default='/home/nhatth/code/datasets/AFORS2022/truongvanminh/data_RGB',
                help='path to RGB data folder.')
ap.add_argument('--of_data_dir', default='/home/nhatth/code/datasets/AFORS2022/truongvanminh/data_OF',
                help='path to OF data folder.')
ap.add_argument('--rgb_data_file', default='/home/nhatth/code/datasets/AFORS2022/truongvanminh/data_RGB.csv',
                help='path to RGB data file.')
ap.add_argument('--of_data_file', default='/home/nhatth/code/datasets/AFORS2022/truongvanminh/data_OF.csv',
                help='path to OF data file.')
ap.add_argument('--rgb_mean_std', default='/home/nhatth/code/datasets/AFORS2022/truongvanminh/data_RGB.pt',
                help='path to RGB mean and std file.')
ap.add_argument('--of_mean_std', default='/home/nhatth/code/datasets/AFORS2022/truongvanminh/data_OF.pt',
                help='path to OF mean and std file.')
ap.add_argument('--label_file', default='/home/nhatth/code/datasets/AFORS2022/truongvanminh/label.txt',
                help='path to label file.')
ap.add_argument('--output_dir', default='outputs/RGB_OF',
                help='path to output folder.')

ap.add_argument('--test_size', type=float, default=0.2,
                help='percentage of test set.')
ap.add_argument('--n_folds', type=int, default=5,
                help='number of k-folds cross-validation.')
ap.add_argument('--seed', type=int, default=42,
                help='seed for random generator')

ap.add_argument('--epochs', type=int, default=25,
                help='number of epochs to train our network for.')
ap.add_argument('--batch_size', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--val_frequency', type=int, default=1,
                help='number of epochs between each validation.')
ap.add_argument('--save_frequency', type=int, default=1,
                help='number of epochs between each checkpoint.')
args = ap.parse_args()

# check cuda
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

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
        return (*multi_stream_input_frames, y)


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

if args.n_folds > 1:
    # kfold = StratifiedKFold(n_splits=args.n_folds)
    kfold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    kfold = kfold.split(np.arange(len(y)), y)
else:
    # train_ind, test_ind = train_test_split(np.arange(len(y)), test_size=args.test_size, random_state=args.seed)
    train_ind, test_ind = train_test_split(np.arange(len(y)), test_size=args.test_size, random_state=args.seed)
    kfold = [(train_ind, test_ind)]

# train, test split
fold_id = 0
for train_ind, test_ind in kfold:
    fold_id += 1

    fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_id}') if args.n_folds > 1 else args.output_dir
    os.makedirs(fold_output_dir, exist_ok=True)
    checkpoint_file = os.path.join(fold_output_dir, 'checkpoint.pt')
    model_file = os.path.join(fold_output_dir, 'model.pt')
    result_file = os.path.join(fold_output_dir, 'result.pt')
    accuracy_figure_file = os.path.join(fold_output_dir, 'accuracy.png')
    loss_figure_file = os.path.join(fold_output_dir, 'loss.png')

    if os.path.exists(result_file):
        print(f'[FOLD {fold_id}/{args.n_folds}] Skipping...')
        continue

    X_RGB_train, X_RGB_test = X_RGB[train_ind], X_RGB[test_ind]
    X_OF_train, X_OF_test = X_OF[train_ind], X_OF[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]
    if args.n_folds > 1:
        print(f'[FOLD {fold_id}/{args.n_folds}]')
    print(f'Training instances: {len(y_train)}')
    print(f'Validataion instances: {len(y_test)}')

    train_set = MultiStreamUCF11([X_RGB_train, X_OF_train], y_train, transform=[rgb_transform, of_transform],
                                 device=device)
    val_set = MultiStreamUCF11([X_RGB_test, X_OF_test], y_test, transform=[rgb_transform, of_transform], device=device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

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

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optim
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )


    # training
    def fit(model, train_loader):
        model.train()
        train_running_loss = 0.0
        train_running_correct = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (X_RGB, X_OF, y) in pbar:
            optimizer.zero_grad()
            outputs = model([X_RGB, X_OF])
            preds = outputs.argmax(1)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_running_correct += (preds == y).sum().item()
            pbar.set_description(f'[Training iter {i + 1}/{len(train_loader)}] batch_loss={loss.item():.03f}')
        train_loss = train_running_loss / len(train_loader.dataset)
        train_accuracy = 100. * train_running_correct / len(train_loader.dataset)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
        return train_loss, train_accuracy


    # validating
    @torch.no_grad()
    def validate(model, val_loader):
        # model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, (X_RGB, X_OF, y) in pbar:
            outputs = model([X_RGB, X_OF])
            preds = outputs.argmax(1)

            loss = criterion(outputs, y)
            val_running_loss += loss.item()
            val_running_correct += (preds == y).sum().item()
            pbar.set_description(f'[Validation iter {i + 1}/{len(val_loader)}] batch_loss={loss.item():.03f}')
        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = 100. * val_running_correct / len(val_loader.dataset)
        print(f'Val loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
        return val_loss, val_accuracy


    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    start = time.time()
    start_epoch = 0

    # resume from previous states
    if os.path.exists(checkpoint_file):
        state_dicts = torch.load(checkpoint_file, map_location=device)
        start_epoch = state_dicts['epoch']
        print(f'Continuing fold {fold_id} from epoch {start_epoch + 1}')
        model.load_state_dict(state_dicts['model_state_dict'])
        optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
        scheduler.load_state_dict(state_dicts['scheduler_state_dict'])

        train_loss = state_dicts['train_loss']
        train_accuracy = state_dicts['train_accuracy']
        val_loss = state_dicts['val_loss']
        val_accuracy = state_dicts['val_accuracy']
        del state_dicts

    for epoch in range(start_epoch, args.epochs):
        print(f"[Epoch {epoch + 1} / {args.epochs}]")

        train_epoch_loss, train_epoch_accuracy = fit(model, train_loader)
        if epoch % args.val_frequency == 0 or epoch == args.epoch - 1:
            val_epoch_loss, val_epoch_accuracy = validate(model, val_loader)

        # print(f'[Epoch {epoch+1} / {args.epochs}] Training summary: train_loss={train_loss:.3f}, train_acc={train_accuracy:.3f}')
        # print(f'[Epoch {epoch+1} / {args.epochs}] Validation summary: val_loss={val_loss:.3f}, val_acc={val_accuracy:.3f}')

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        scheduler.step(val_epoch_loss)

        # save state dicts after each epoch
        if epoch % args.save_frequency == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, checkpoint_file)
            print(f'\t[Epoch {epoch + 1} / {args.epochs}] Model checkpointed at {checkpoint_file}')
        print()

    end = time.time()
    print(f'TRAINING COMPLETED! Total elapsed time: {(end - start) / 60:.3f} minutes')

    # save results
    torch.save({
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, result_file)

    # plot accuracy
    plt.figure(figsize=(10, 7))
    px = np.arange(len(train_accuracy)) + 1
    plt.plot(px, train_accuracy, color='green', label='train accuracy')
    plt.plot(px, val_accuracy, color='blue', label='validataion accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(1, args.epochs)
    plt.grid()
    plt.legend()
    plt.savefig(accuracy_figure_file, bbox_inches='tight', transparent=True)

    # plot loss
    plt.figure(figsize=(10, 7))
    px = np.arange(len(train_loss)) + 1
    plt.plot(px, train_loss, color='orange', label='train loss')
    plt.plot(px, val_loss, color='red', label='validataion loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(1, args.epochs)
    plt.grid()
    plt.legend()
    plt.savefig(loss_figure_file, bbox_inches='tight', transparent=True)

    # serialize the model to disk
    torch.save(model.state_dict(), model_file)

# Assemble kfold results
if args.n_folds > 1:
    result_file = os.path.join(args.output_dir, 'result.pt')
    accuracy_figure_file = os.path.join(args.output_dir, 'accuracy.png')
    loss_figure_file = os.path.join(args.output_dir, 'loss.png')

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for fold_id in range(1, args.n_folds + 1):
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_id}') if args.n_folds > 1 else args.output_dir
        fold_result_file = os.path.join(fold_output_dir, 'result.pt')
        fold_result = torch.load(fold_result_file)
        train_accuracies.append(fold_result['train_accuracy'])
        val_accuracies.append(fold_result['val_accuracy'])
        train_losses.append(fold_result['train_loss'])
        val_losses.append(fold_result['val_loss'])

    train_accuracies = np.array(train_accuracies)
    val_accuracies = np.array(val_accuracies)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # save results
    torch.save({
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, result_file)

    # plot accuracy
    plt.figure(figsize=(10, 7))
    px = np.arange(train_accuracies.shape[1]) + 1
    plt.plot(px, train_accuracies.mean(0), color='green', zorder=3, label='train accuracies')
    plt.fill_between(px, train_accuracies.min(0), train_accuracies.max(0), facecolor='green', alpha=0.2, zorder=1)
    plt.plot(val_accuracies.mean(0), color='blue', zorder=3, label='validataion accuracies')
    plt.fill_between(px, val_accuracies.min(0), val_accuracies.max(0), facecolor='blue', alpha=0.2, zorder=1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(1, args.epochs)
    plt.grid()
    plt.legend()
    plt.savefig(accuracy_figure_file, bbox_inches='tight', transparent=True)

    # plot loss
    plt.figure(figsize=(10, 7))
    px = np.arange(train_losses.shape[1]) + 1
    plt.plot(px, train_losses.mean(0), color='orange', zorder=3, label='train losses')
    plt.fill_between(px, train_losses.min(0), train_losses.max(0), facecolor='orange', alpha=0.2, zorder=1)
    plt.plot(px, val_losses.mean(0), color='red', zorder=3, label='validataion losses')
    plt.fill_between(px, val_losses.min(0), val_losses.max(0), facecolor='red', alpha=0.2, zorder=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(1, args.epochs)
    plt.grid()
    plt.legend()
    plt.savefig(loss_figure_file, bbox_inches='tight', transparent=True)
