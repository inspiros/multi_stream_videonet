import torch
import torch.nn as nn
from datasets import loso_har_feature
from models.parallel_module_list import ParallelModuleList
from models.transformer import MutualMultiheadAttention, DEQMutualMultiheadAttention
from models.fusion import FusionBlock
from models.deq import DEQBlock, DEQSequential


class MultistreamDataset(nn.Module):
    def __init__(self,
                 num_streams,
                 num_classes,
                 in_features,
                 transfer_features,
                 use_deq=False):
        super(MultistreamDataset, self).__init__()
        self.use_deq = use_deq

        self.stem = ParallelModuleList([
            nn.Sequential(
                nn.Linear(in_features, transfer_features),
                nn.ReLU()
            )
            for _ in range(num_streams)
        ])

        if use_deq:
            self.transfer = DEQBlock(
                f=DEQMutualMultiheadAttention(
                    num_streams=num_streams,
                    in_channels=transfer_features
                ),
            )
        else:
            self.transfer = MutualMultiheadAttention(
                num_streams=num_streams,
                in_channels=transfer_features)

        self.fuse = FusionBlock(num_streams=num_streams,
                                in_channels=transfer_features)
        self.clf_fc = nn.Linear(transfer_features, num_classes)

    def forward(self, xs):
        xs = self.stem(xs)
        xs = self.transfer(xs)
        x = self.fuse(xs)
        x = self.clf_fc(x)
        return x


def main():
    dataset = loso_har_feature(root='D:/datasets/har_feature/ixmas/c3d_bn')

    Xs_train, y_train, Xs_test, y_test = dataset.where(test_actor='alba')[:]
    print(Xs_train[0].shape, Xs_test[0].shape)
    model = MultistreamDataset(num_streams=len(Xs_train),
                               num_classes=len(y_train.unique()),
                               in_features=Xs_train[0].size(1),
                               transfer_features=256,
                               use_deq=False)
    out = model(Xs_test)
    print(out)


if __name__ == '__main__':
    main()
