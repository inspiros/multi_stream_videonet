import torch
from torch.utils.data import DataLoader, Dataset


class MultiStreamDataset(Dataset):
    def __init__(self, num_streams, num_classes, size, num_samples=1000, device='cpu'):
        self.num_streams = num_streams
        self.num_classes = num_classes
        self.size = size
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        Xs = []
        for stream_id in range(self.num_streams):
            Xs.append(torch.rand(*self.size, device=self.device))
        y = torch.randint(0, self.num_classes, (1,))
        return (*Xs, y)


if __name__ == '__main__':
    dataset = MultiStreamDataset(2, 10, (3, 16, 112, 112))
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    for X1, X2, y in train_loader:
        print(X1.shape, X2.shape, y.shape)
