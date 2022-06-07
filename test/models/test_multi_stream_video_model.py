import torch

from models.videos.multi_stream import get_model


def test_multi_stream_r2plus1d_18():
    num_streams = 2
    model = get_model(
        'multi_stream_r2plus1d_18',
        num_classes=10,
        num_streams=num_streams,
        transfer_stages=[3, 4],
        fusion_stage=5,
    )

    xs = [torch.randn(1, 3, 32, 224, 224) for _ in range(num_streams)]
    print('xs', [xs[stream_id].shape for stream_id in range(num_streams)])
    y = model(xs)
    print('y', y.shape)


def test_multi_stream_movinet():
    num_streams = 2
    model = get_model(
        'multi_stream_movinet_a0',
        num_classes=10,
        num_streams=num_streams,
        transfer_stages=[4, 5, 6],
        fusion_stage=8,
    )

    xs = [torch.randn(1, 3, 32, 224, 224) for _ in range(num_streams)]
    print('xs', [xs[stream_id].shape for stream_id in range(num_streams)])
    y = model(xs)
    print('y', y.shape)


if __name__ == '__main__':
    # test_multi_stream_r2plus1d_18()
    test_multi_stream_movinet()
