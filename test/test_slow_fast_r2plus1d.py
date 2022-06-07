import time
import torch

from models.videos.slow_fast_r2plus1d import slow_fast_r2plus1d_18


def main():
    model = slow_fast_r2plus1d_18(num_classes=10,
                                  pretrained=True)

    x = torch.rand(2, 3, 16, 112, 112)

    start = time.time()
    y = model(x)
    forward_time = time.time()
    print('forward_time', forward_time - start)
    print('output', y.shape)


if __name__ == '__main__':
    main()
