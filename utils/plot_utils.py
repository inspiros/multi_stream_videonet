import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

__all__ = ['plot_metrics',
           'plot_confusion_matrix']


def plot_metrics(train_history, test_history, label='', save_file=None):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    for i, (history, part) in enumerate(zip([train_history, test_history], ['train', 'test'])):
        epochs = np.array(list(history.keys()))
        vals = np.array(list(history.values()))
        if vals.ndim == 1:
            ax.plot(epochs, vals, zorder=3, label=f'{part} {label}')
        elif vals.ndim == 2:
            ax.plot(epochs, vals.mean(1), zorder=3, label=f'{part} {label}')
            ax.fill_between(epochs, vals.min(1), vals.max(1), alpha=0.2, zorder=1)
        else:
            raise RuntimeError(f'Cannot plot {vals.ndim}-D data.')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.set_xlim(1, max(max(train_history.keys()), max(test_history.keys())))
    ax.grid()
    ax.legend()

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, bbox_inches='tight', transparent=True)
    return fig, ax


def plot_confusion_matrix(cm, class_names=None, save_file=None, **kwargs):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    classes = class_names if class_names is not None else np.arange(len(cm))
    heatmap_kwargs = dict(
        vmin=0,
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        fmt='d',
    )
    sn.heatmap(cm,
               **{**heatmap_kwargs, **kwargs},
               ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, bbox_inches='tight', transparent=True)
    return fig, ax
