from models.videos import get_model


def main():
    model = get_model('mc3d_16', num_classes=10)


if __name__ == '__main__':
    main()
