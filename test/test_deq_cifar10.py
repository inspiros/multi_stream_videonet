import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import DEQBlock
from models.videos.multi_stream.parallel_modules import ParallelModuleList
import argparse


class DEQConv(nn.Module):
    def __init__(self, in_channels, num_groups=8):
        super(DEQConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.norm2 = nn.GroupNorm(num_groups, in_channels)
        self.act = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = self.norm1(self.act(self.conv1(z)))
        y = self.norm2(self.act(x + self.conv2(y)))
        return y


class DEQBatchNorm(nn.Module):
    def __init__(self, num_streams, in_channels):
        super(DEQBatchNorm, self).__init__()
        self.bn = nn.Sequential(
            ParallelModuleList(
                [nn.GELU() for _ in range(num_streams)]
            ),
            ParallelModuleList(
                [nn.BatchNorm2d(in_channels) for _ in range(num_streams)]
            ),
        )

    def forward(self, zs, xs):
        return self.bn(zs)


class MultiStreamDEQModel(nn.Module):
    def __init__(self,
                 num_classes,
                 deq_mode='safe_backward_hook',
                 deq_num_layers=6,
                 deq_max_iters=20):
        super(MultiStreamDEQModel, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5)),
            nn.Conv2d(16, 32, kernel_size=(5, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.f = DEQBlock(
            f=DEQConv(in_channels=64),
            deq_mode=deq_mode,
            num_layers=deq_num_layers,
            max_iters=deq_max_iters,
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(64, num_classes)

    def set_deq_mode(self, deq_mode):
        for m in self.modules():
            if isinstance(m, DEQBlock):
                m.set_deq_mode(deq_mode)

    def forward(self, x):
        x = self.stem(x)
        x = self.f(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


def test_train(args):
    torch.manual_seed(args.seed)
    cifar10_train = datasets.CIFAR10(root="data/cifar10",
                                     train=True,
                                     download=True,
                                     transform=transforms.ToTensor())
    cifar10_test = datasets.CIFAR10(root="data/cifar10",
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())
    train_loader = DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = MultiStreamDEQModel(num_classes=10,
                                deq_mode=args.deq_mode,
                                deq_num_layers=args.deq_num_layers,
                                deq_max_iters=args.deq_max_iters,
                                ).to(args.device)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

    init_weights_path = '../weights/deq_cifar10.pt'
    if os.path.exists(init_weights_path):
        model.load_state_dict(torch.load(init_weights_path, map_location=args.device))
    else:
        torch.save(model.state_dict(), init_weights_path)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                           args.max_epochs * len(train_loader),
                                                           eta_min=1e-6)

    # standard training or evaluation loop
    def epoch(epoch_id, loader, model, optim=None, lr_scheduler=None):
        epoch_loss, epoch_acc = 0., 0.
        model.eval() if optim is None else model.train()
        task = 'Training' if optim is not None else 'Testing'
        pbar = tqdm(enumerate(loader), desc=f'[Epoch {epoch_id + 1}] ({task})')
        for batch_id, (X, y) in pbar:
            X, y = X.to(args.device), y.to(args.device)
            yp = model(X)
            loss = criterion(yp, y)
            if optim is not None:
                optim.zero_grad()
                loss.backward()
                optim.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

            batch_acc = yp.argmax(1).eq(y).sum().item() / X.size(0)
            batch_loss = loss.item()
            epoch_acc += batch_acc * X.size(0)
            epoch_loss += batch_loss * X.size(0)

            pbar.set_description(f'[Epoch {epoch_id + 1} - Iter {batch_id + 1}/{len(loader)}] ({task}) '
                                 f'acc={batch_acc:.03f}, loss={batch_loss:.03f}')
        return epoch_acc / len(loader.dataset), epoch_loss / len(loader.dataset)

    results = []
    print('Training DEQ model')
    for epoch_id in range(args.max_epochs):
        if args.deq_mode:
            if epoch_id < args.deq_init_epochs:
                model.set_deq_mode('deterministic')
            else:
                model.set_deq_mode('safe_backward_hook')
        else:
            model.set_deq_mode('deterministic')
        train_err, train_loss = epoch(epoch_id, train_loader, model, optim, scheduler)
        test_err, test_loss = epoch(epoch_id, test_loader, model)
        print(f'[Epoch {epoch_id + 1}/{args.max_epochs}] '
              f'train_acc={train_err:.03f}, train_loss={train_loss:.03f}, '
              f'test_acc={test_err:.03f}, test_loss={test_loss:.03f}')
        results.append([train_err, train_loss, test_err, test_loss])

    results = np.array(results)
    print()
    print(results)
    if args.save_file is not None:
        os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
        np.savetxt(args.save_file, results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--deq_mode', action='store_true')
    parser.add_argument('--deq_num_layers', type=int, default=6)
    parser.add_argument('--deq_max_iters', type=int, default=25)
    parser.add_argument('--deq_init_epochs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_file', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    test_train(parse_args())
