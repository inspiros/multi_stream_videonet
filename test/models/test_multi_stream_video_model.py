import torch

from models.videos.multi_stream import get_model


def main():
    num_streams = 2
    model = get_model(
        'multi_stream_movinet_a0',
        num_classes=10,
        num_streams=num_streams,
        transfer_stages=[4, 5, 6],
        fusion_stage=7,
    )
    print(model)

    xs = [torch.randn(1, 3, 32, 224, 224) for _ in range(num_streams)]
    y = model(xs)
    print(y.shape)


if __name__ == '__main__':
    main()
