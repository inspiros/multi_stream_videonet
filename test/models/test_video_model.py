import torch

from models.videos import get_model


def main():
    model = get_model('movinet_a0', num_classes=10)
    print(model)

    x = torch.randn(1, 3, 32, 224, 224)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
