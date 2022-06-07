import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import DEQBlock, DEQMutualMultiheadNonlocal2d, MutualMultiheadNonlocal2d
from models.videos.multi_stream.fusion import FusionBlock
from models.videos.multi_stream.parallel_module_list import ParallelModuleList

device = torch.device("cuda:0")


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
                 num_streams,
                 num_classes,
                 deq_channels=64,
                 deq_mode='safe_backward_hook',
                 deq_num_layers=6,
                 deq_max_iters=20):
        super(MultiStreamDEQModel, self).__init__()
        self.num_streams = num_streams

        # self.stem = ParallelModuleList([
        #     nn.Conv2d(3, 32, kernel_size=(3, 3))
        #     for _ in range(num_streams)
        # ])
        self.stem = ParallelModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=(3, 3)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(3, 3)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(2, 2)),
                nn.Conv2d(32, deq_channels, kernel_size=(3, 3)),
                nn.BatchNorm2d(deq_channels),
                nn.ReLU()
            ) for _ in range(num_streams)
        ])
        # self.transfer = DEQBlock(
        #     f=DEQSequential(
        #         DEQMutualMultiheadNonlocal2d(
        #             num_streams=num_streams,
        #             in_channels=deq_channels,
        #             hidden_channels=deq_channels // 2,
        #             num_heads=1,
        #         ),
        #         DEQBatchNorm(num_streams,
        #                      deq_channels)
        #     ),
        #     deq_mode=deq_mode,
        #     num_layers=deq_num_layers,
        #     max_iters=deq_max_iters,
        # )
        self.transfer = nn.Sequential(
            MutualMultiheadNonlocal2d(
                num_streams=num_streams,
                in_channels=deq_channels,
                hidden_channels=deq_channels // 2,
                num_heads=1,
            ),
            ParallelModuleList(
                [nn.GELU() for _ in range(num_streams)]
            ),
            ParallelModuleList(
                [nn.BatchNorm2d(deq_channels) for _ in range(num_streams)]
            ),
        )
        self.fuse = FusionBlock(num_streams=num_streams,
                                in_channels=deq_channels)
        self.pool = ParallelModuleList(
            # [nn.AvgPool2d(kernel_size=(3, 3)) for _ in range(num_streams)]
            [nn.AdaptiveAvgPool2d(output_size=(1, 1)) for _ in range(num_streams)]
        )
        self.fc = nn.Linear(deq_channels, num_classes)

    def set_deq_mode(self, deq_mode):
        for m in self.modules():
            if isinstance(m, DEQBlock):
                m.set_deq_mode(deq_mode)

    def forward(self, xs):
        xs = self.stem(xs)
        xs = self.transfer(xs)
        # xs = self.bn(xs)
        xs = self.pool(xs)
        x = self.fuse(xs).flatten(1)
        x = self.fc(x)
        return x


def test_train():
    cifar10_train = datasets.CIFAR10(root="data/cifar10",
                                     train=True,
                                     download=True,
                                     transform=transforms.ToTensor())
    cifar10_test = datasets.CIFAR10(root="data/cifar10",
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())
    train_loader = DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=1)

    model = MultiStreamDEQModel(num_streams=2,
                                num_classes=10,
                                # deq_mode=None,
                                # deq_num_layers=1,
                                ).to(device)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

    max_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                           max_epochs * len(train_loader),
                                                           eta_min=1e-6)

    # standard training or evaluation loop
    def epoch(epoch_id, loader, model, optim=None, lr_scheduler=None):
        epoch_loss, epoch_acc = 0., 0.
        model.eval() if optim is None else model.train()
        task = 'Training' if optim is not None else 'Testing'
        pbar = tqdm(enumerate(loader), desc=f'[Epoch {epoch_id + 1}] ({task})')
        for batch_id, (X, y) in pbar:
            X, y = X.to(device), y.to(device)
            yp = model([X, torch.flip(X, dims=(-1,))])
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

    print('Training DEQ model')
    for epoch_id in range(max_epochs):
        # model.set_deq_mode('autograd')
        if epoch_id < 3:
            model.set_deq_mode('deterministic')
        else:
            model.set_deq_mode('autograd')
        train_err, train_loss = epoch(epoch_id, train_loader, model, optim, scheduler)
        test_err, test_loss = epoch(epoch_id, test_loader, model)
        print(f'[Epoch {epoch_id + 1}/{max_epochs}] '
              f'train_acc={train_err:.03f}, train_loss={train_loss:.03f}, '
              f'test_acc={test_err:.03f}, test_loss={test_loss:.03f}')

    # torch.save(model.state_dict(), 'multi_stream_deq.pt')


def test_forward_backward():
    model = DEQBlock(
        f=DEQMutualMultiheadNonlocal2d(
            num_streams=2,
            in_channels=32,
            hidden_channels=64,
            num_heads=1,
            kernel_size=(3, 3),
        ),
        deq_mode='autograd',
        max_iters=20,
    ).to(device)
    print(model)

    xs = [torch.rand(2, 32, 16, 16, device=device, requires_grad=True)
          for _ in range(model.f.num_streams)]
    ys = model(xs)
    print('output', [_.shape for _ in ys])

    sum(ys).sum().backward()
    print('grad', [_.grad for _ in xs])


if __name__ == '__main__':
    # test_forward_backward()
    test_train()
