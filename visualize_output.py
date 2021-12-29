import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, InsetPosition


def mark_inset(parent_axes, inset_axes, locs=((1, 1), (2, 2), (3, 3), (4, 4)), **kwargs):
    from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    if 'fill' in kwargs:
        pp = BboxPatch(rect, **kwargs)
    else:
        fill = bool({'fc', 'facecolor', 'color'}.intersection(kwargs))
        pp = BboxPatch(rect, fill=fill, **kwargs)
    parent_axes.add_patch(pp)

    connector_defaults = dict(linestyle='dashed')
    cps = []
    for loc1, loc2 in locs:
        p = BboxConnector(inset_axes.bbox, rect,
                          loc1=loc1, loc2=loc2, **{**connector_defaults, **kwargs})
        parent_axes.add_patch(p)
        p.set_clip_on(False)
        cps.append(p)

    return pp, cps


def main():
    output_dir = 'output'

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    zoom_axes = [zoomed_inset_axes(ax, 3, loc='lower center') for ax in axes]
    cmap = get_cmap('Spectral', len(os.listdir(output_dir)))

    results_list = []
    for i, f in enumerate(sorted(os.listdir(output_dir))):
        results = np.loadtxt(os.path.join(output_dir, f))
        results_list.append(results)
        label = os.path.splitext(f)[0]
        axes[0].plot(results[:, 0], label=label, color=cmap(i), linestyle='-')
        axes[1].plot(results[:, 2], label=label, color=cmap(i), linestyle='--')
        zoom_axes[0].plot(results[:, 0], linewidth=2, color=cmap(i), label=label, zorder=3)
        zoom_axes[1].plot(results[:, 2], linewidth=2, color=cmap(i), label=label, zorder=3)
    results_list = np.stack(results_list)

    for i, (ax, zoom_ax) in enumerate(zip(axes, zoom_axes)):
        zoom_ax.set_xlim(25, 29)
        if i == 0:
            zoom_ax.set_ylim(0.85, 0.92)
        elif i == 1:
            zoom_ax.set_ylim(0.75, 0.8)
        # mark_inset(ax,
        #            zoom_ax,
        #            facecolor=(0.9, 0.9, 0.9),
        #            edgecolor=(0, 0, 0),
        #            linewidth=1)
        zoom_ax.set_xticks([])
    axes[0].set_title('Train accuracy')
    axes[1].set_title('Test accuracy')
    axes[0].legend()
    axes[1].legend()
    plt.show()


if __name__ == '__main__':
    main()
