import torch
from tqdm import tqdm

__all__ = ['ClassificationTrainer']


def _default_unpack_data(batch_data, dtype=None, device=None):
    X = [_.to(dtype=dtype, device=device) for _ in batch_data[:-1]]
    if len(X) == 1:
        X = X[0]
    return X


def _default_unpack_label(batch_data, dtype=None, device=None):
    y = batch_data[-1].to(dtype=dtype, device=device)
    return y


class ClassificationTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 device='cpu',
                 data_dtype=torch.float32,
                 label_dtype=torch.long,
                 unpack_data_func=None,
                 unpack_label_func=None,
                 verbose=True):
        self.device = torch.device(device)
        self.data_dtype = data_dtype
        self.label_dtype = label_dtype

        self.model = model.to(dtype=self.data_dtype, device=self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.unpack_data_func = unpack_data_func if unpack_data_func is not None else _default_unpack_data
        self.unpack_label_func = unpack_label_func if unpack_label_func is not None else _default_unpack_label
        self.verbose = verbose

    def fit(self, train_loader):
        self.model.train()

        running_loss = 0.0
        running_correct = 0
        pbar = enumerate(train_loader)
        if self.verbose:
            pbar = tqdm(pbar, total=len(train_loader), desc='Training')
        for batch_id, batch_data in pbar:
            X = self.unpack_data_func(batch_data, dtype=self.data_dtype, device=self.device)
            y = self.unpack_label_func(batch_data, dtype=self.label_dtype, device=self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            preds = outputs.argmax(1)

            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_correct += preds.eq(y).sum().item()
            if self.verbose:
                pbar.set_description(f'[Training iter {batch_id + 1}/{len(train_loader)}]'
                                     f' batch_loss={loss.item():.03f}')
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_correct / len(train_loader.dataset)
        return train_loss, train_accuracy

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        pbar = enumerate(val_loader)
        if self.verbose:
            pbar = tqdm(pbar, total=len(val_loader), desc='Validation')
        for batch_id, batch_data in pbar:
            X = self.unpack_data_func(batch_data, dtype=self.data_dtype, device=self.device)
            y = self.unpack_label_func(batch_data, dtype=self.label_dtype, device=self.device)

            outputs = self.model(X)
            preds = outputs.argmax(1)

            loss = self.criterion(outputs, y)
            acc = preds.eq(y).float().mean()

            running_loss += loss.item()
            running_correct += preds.eq(y).sum().item()
            if self.verbose:
                pbar.set_description(f'[Validation iter {batch_id + 1}/{len(val_loader)}]'
                                     f' batch_loss={loss.item():.03f}'
                                     f' batch_acc={acc.item():.02%}')
        val_loss = running_loss / len(val_loader.dataset)
        val_accuracy = running_correct / len(val_loader.dataset)
        return val_loss, val_accuracy

    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()

        y_trues = []
        y_preds = []
        pbar = enumerate(test_loader)
        if self.verbose:
            pbar = tqdm(pbar, total=len(test_loader), desc='Testing')
        for batch_id, batch_data in pbar:
            X = self.unpack_data_func(batch_data, dtype=self.data_dtype, device=self.device)
            y = self.unpack_label_func(batch_data, dtype=self.label_dtype, device='cpu')

            outputs = self.model(X)
            preds = outputs.argmax(1).cpu()

            acc = preds.eq(y).float().mean()

            y_trues.append(y)
            y_preds.append(preds)
            if self.verbose:
                pbar.set_description(f'[Testing iter {batch_id + 1}/{len(test_loader)}]'
                                     f' batch_acc={acc.item():.02%}')
        y_trues = torch.cat(y_trues)
        y_preds = torch.cat(y_preds)
        return y_trues, y_preds
