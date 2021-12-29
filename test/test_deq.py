import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import DEQBlock

device = torch.device(f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu")


class ResNetLayer(nn.Module):

    def __init__(self, n_channels, n_inner_channels, kernel_size, num_groups=8):
        super(ResNetLayer, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.act = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = self.norm1(self.act(self.conv1(z)))
        y = self.norm3(self.act(z + self.norm2(x + self.conv2(y))))
        return y


class DEQModel(nn.Module):
    def __init__(self, num_classes, deq_channels, deq_inner_channels):
        super(DEQModel, self).__init__()

        self.stem = nn.Sequential(
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
        )
        self.deq = DEQBlock(f=ResNetLayer(deq_channels, deq_inner_channels, kernel_size=3),
                            solver='anderson',
                            max_iters=30)

        self.bn = nn.BatchNorm2d(deq_channels)
        self.pool = nn.AvgPool2d(kernel_size=(3, 3))
        self.fc = nn.Linear(deq_channels * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.deq(x)
        x = self.bn(self.nl(x))
        x = self.pool(x).flatten(1)
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
    train_loader = DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=8)
    test_loader = DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=8)

    model = DEQModel(num_classes=10,
                     deq_channels=48,
                     deq_inner_channels=64).to(device)

    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

    max_epochs = 50
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
        pbar = tqdm(enumerate(loader), desc=f'[Epoch {epoch_id}] ({task})')
        for batch_id, (X, y) in pbar:
            X, y = X.to(device), y.to(device)
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

            pbar.set_description(f'[Epoch {epoch_id} - Iter {batch_id}/{len(loader)}] ({task}) '
                                 f'acc={batch_acc:.03f}, loss={batch_loss:.03f}')
        return epoch_acc / len(loader.dataset), epoch_loss / len(loader.dataset)

    print('Training DEQ model')
    for epoch_id in range(max_epochs):
        train_err, train_loss = epoch(epoch_id, train_loader, model, optim, scheduler)
        test_err, test_loss = epoch(epoch_id, test_loader, model)
        print(f'[Epoch {epoch_id + 1}/{max_epochs}] '
              f'train_acc={train_err:.03f}, train_loss={train_loss:.03f}, '
              f'test_acc={test_err:.03f}, test_loss={test_loss:.03f}')

    torch.save(model.state_dict(), 'deq.pt')


def test_single_output_layer():
    class EquilibriumLayer(nn.Module):
        def __init__(self, channels, kernel_size=3):
            super(EquilibriumLayer, self).__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)
            # self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)

        def forward(self, z, x):
            y = self.bn1(self.conv1(z).relu()) + x
            y = self.bn2(self.conv2(y).relu())
            return y

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size=(1, 1)),
        DEQBlock(f=EquilibriumLayer(2, kernel_size=3),
                 solver='anderson',
                 max_iters=30),
    )
    model.to(dtype=torch.float64, device=device)
    model.train()
    optim = torch.optim.Optimizer(model.parameters(), dict())
    print(model)

    x = torch.rand(1, 2, 4, 4, dtype=torch.float64, device=device, requires_grad=True)

    model[1].set_deq_mode(deq_mode='backward_hook')
    y_old = model(x).sum()
    y_old.backward()
    x_grad_old = x.grad.clone()
    x.grad.data.fill_(0)
    # print('conv', model[1].f.conv1.weight.grad)
    print('conv', model[0].weight.grad)
    optim.zero_grad()
    # print(y)
    # print(x_grad_old)
    # print()

    model[1].set_deq_mode(deq_mode='autograd')
    y = model(x).sum()
    y.backward()
    x_grad = x.grad.clone()
    x.grad.data.fill_(0)
    # print('conv', model[1].f.conv1.weight.grad)
    print('conv', model[0].weight.grad)
    optim.zero_grad()
    # print(y)
    # print(x_grad)
    # print()

    print('y_diff', y_old - y)
    print('x_grad_diff', x_grad_old - x_grad)

    # exit()
    # true_grad = gradcheck(lambda _: model(_).sum(), inputs=(x,))
    # print('grad_check', true_grad)


def test_multi_output_layer():
    class MultiStreamEquilibriumLayer(nn.Module):
        def __init__(self, streams, channels, kernel_size=3):
            super(MultiStreamEquilibriumLayer, self).__init__()
            self.streams = streams
            self.conv1 = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)] * streams)
            self.conv2 = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)] * streams)
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(channels)] * streams)
            self.bn2 = nn.ModuleList([nn.BatchNorm2d(channels)] * streams)
            # self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)

        def forward(self, zs, xs):
            ys = [self.bn1[stream_id](self.conv1[stream_id](zs[stream_id]).relu())
                  for stream_id in range(self.streams)]
            ys = [ys[stream_id] + xs[stream_id] for stream_id in range(self.streams)]
            ys = [self.bn2[stream_id](self.conv2[stream_id](ys[stream_id]).relu())
                  for stream_id in range(self.streams)]
            return ys

    torch.manual_seed(0)
    streams = 2
    model = DEQBlock(f=MultiStreamEquilibriumLayer(streams=streams,
                                                   channels=4,
                                                   kernel_size=3),
                     solver='anderson',
                     max_iters=30)
    model.to(dtype=torch.float64, device=device)
    model.train()
    print(model)

    xs = [torch.rand(1, 4, 6, 6, dtype=torch.float64, device=device, requires_grad=True)
          for _ in range(streams)]
    print([_.shape for _ in xs])

    y = sum(model(xs)).sum()
    y.backward()
    print([_.grad for _ in xs])


if __name__ == '__main__':
    test_single_output_layer()
    # test_multi_output_layer()
    # test_train()
