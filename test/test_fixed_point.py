import numpy as np
import matplotlib.pyplot as plt


def main():
    def f(x, max_iters=1, tol=1e-5):
        z = x / 2 + 1
        if max_iters == 0 or np.all(np.abs(z - x) <= tol):
            return z, 1
        return f(z, max_iters - 1, tol), 1

    print(f(x=3, max_iters=100))
    print(f(x=-49999, max_iters=100))


def sparse_attention_mask():
    dims = (5, 5, 5)
    kernel = (3, 3, 5)
    n_positions = np.prod(dims)

    def index_to_coord(index, dims):
        coord = []
        for i in reversed(range(len(dims))):
            coord.append(int((index // np.prod(dims[i + 1:])) % dims[i]))
        return np.array(list(reversed(coord)))

    mask = np.zeros((n_positions, n_positions))
    dist = np.array(kernel) // 2
    for i in range(mask.shape[0]):
        coord_i = index_to_coord(i, dims)
        for j in range(mask.shape[1]):
            coord_j = index_to_coord(j, dims)
            if np.all(coord_i == coord_j):
                mask[i, j] = 1
            elif np.all(np.abs(coord_j - coord_i) <= dist):
                mask[i, j] = 0.6
    print(mask)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='YlGn', extent=(0, n_positions, n_positions, 0))
    # plt.gca().invert_yaxis()
    plt.gca().grid()
    plt.gca().set_xticks(np.arange(n_positions))
    plt.gca().set_yticks(np.arange(n_positions))

    plt.gca().tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    plt.savefig('../images/sparse_attention_mask.png', bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    sparse_attention_mask()
