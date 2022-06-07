import torch

from models.transformer.non_local import MultiheadNonlocal2d, MutualMultiheadNonlocal2d


def test_nonlocal():
    m = MultiheadNonlocal2d(in_channels=16,
                            num_heads=2,
                            attn_type='nystrom',
                            p_landmark=0.25)

    x = torch.rand(1, 16, 12, 12)
    print('x', x.shape)
    y = m(x)
    print('y', y.shape)


def test_mutual_nonlocal():
    m = MutualMultiheadNonlocal2d(
        num_streams=2,
        in_channels=16,
        num_heads=2,
        attn_type='nystrom',
        p_landmark=0.25,
    )

    xs = [torch.rand(1, 16, 12, 12) for _ in range(m.num_streams)]
    print('xs', [xs[stream_id].shape for stream_id in range(len(xs))])
    ys = m(xs)
    print('ys', [ys[stream_id].shape for stream_id in range(len(ys))])


if __name__ == '__main__':
    # test_mha()
    # test_mutual_mha()
    # test_nonlocal()
    test_mutual_nonlocal()
