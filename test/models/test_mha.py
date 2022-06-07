import torch

from models.transformer.mha import MultiheadAttention, MutualMultiheadAttention


def test_mha():
    m = MultiheadAttention(in_features=128,
                           attn_type='nystrom',
                           p_landmark=0.5)

    x = torch.rand(1, 10, 128)
    print('x', x.shape)
    y = m(x)
    print('y', y.shape)


def test_mutual_mha():
    m = MutualMultiheadAttention(
        num_streams=2,
        in_features=128,
        attn_type='nystrom',
        p_landmark=0.5,
    )

    xs = [torch.rand(1, 10, 128) for _ in range(m.num_streams)]
    print('xs', [xs[stream_id].shape for stream_id in range(len(xs))])
    ys = m(xs)
    print('ys', [ys[stream_id].shape for stream_id in range(len(ys))])


if __name__ == '__main__':
    # test_mha()
    test_mutual_mha()
