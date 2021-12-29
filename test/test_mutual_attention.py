import torch

from models import MultiheadNonlocal3d, MutualMultiheadNonlocal3d

if __name__ == '__main__':
    # model = MultiheadNonlocal3d(in_channels=256,
    #                             num_heads=4,
    #                             kernel_size=(1, 2, 2),
    #                             )
    # x = torch.rand(2, 256, 4, 28, 28)
    # y = model(x)
    # print(y.shape)

    model = MutualMultiheadNonlocal3d(num_streams=2,
                                      in_channels=256,
                                      hidden_channels=64,
                                      num_heads=1,
                                      kernel_size=(1, 3, 3),
                                      )
    xs = [torch.rand(2, 256, 4, 28, 28) for _ in range(model.num_streams)]
    ys = model(xs)
    print([_.shape for _ in ys])
