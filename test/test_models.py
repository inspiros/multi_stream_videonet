from models.r2plus1d import r2plus1d_18
import torch


def main():
    device = torch.device('cpu')

    model = r2plus1d_18().to(device)
    x = torch.rand(1, 3, 32, 224, 224, device=device)
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    main()
