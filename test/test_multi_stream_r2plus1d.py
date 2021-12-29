import time
import torch

from models import multi_stream_r2plus1d_18


def main():
    # model = r2plus1d_18()
    # x = torch.rand(2, 3, 16, 224, 224)
    # print(model(x).shape)
    # exit()

    num_streams = 2
    model = multi_stream_r2plus1d_18(num_classes=10,
                                     num_streams=num_streams,
                                     fusion_stage=3,
                                     transfer_stages=(3,),
                                     pretrained=True)

    xs = [torch.rand(2, 3, 16, 112, 112)
          for _ in range(num_streams)]

    start = time.time()
    y = model(xs)
    forward_time = time.time()
    print('forward_time', forward_time - start)

    y.sum().backward()
    backward_time = time.time()
    print('forward_time', backward_time - forward_time)

    print('output', y.shape)


if __name__ == '__main__':
    main()
