import torch
from sklearn import model_selection


def train_test_split(X, y, test_size=0.2, shuffle=True):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return X_train, y_train, X_test, y_test


def multiview_train_test_split(Xs, y, test_size=0.2, shuffle=False):
    test_mask = torch.rand(len(y)).le(test_size)
    train_mask = test_mask.logical_not()
    if shuffle:
        train_mask = train_mask[torch.randperm(len(train_mask))]
        test_mask = test_mask[torch.randperm(len(test_mask))]
    Xs_train = [_[train_mask] for _ in Xs]
    Xs_test = [_[test_mask] for _ in Xs]
    y_train = y[train_mask]
    y_test = y[test_mask]
    return Xs_train, y_train, Xs_test, y_test
