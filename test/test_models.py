import torch

from models.videos.c3d import c3d_bn


def main():
    device = torch.device('cpu')

    model = c3d_bn(temporal_slice=32).to(device)
    x = torch.rand(1, 3, 32, 112, 112, device=device)
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    main()
