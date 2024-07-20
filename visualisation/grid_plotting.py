import matplotlib.pyplot as plt
import numpy as np


def hist_grid(samples, height: int, width: int, figsize=(15,15), x_lims: tuple = None, n_bins=50, x_label='Reward',
              title=None, shared_x=True):

    if x_lims is None:
        if shared_x:
            x_lims = (samples.min(), samples.max())
            bins = np.linspace(x_lims[0], x_lims[1], n_bins)
        else:
            bins = n_bins
    else:
        assert len(x_lims) == 2, "x_lims must be a tuple of length 2"
        assert shared_x, "x_lims only supported for shared_x=True"
        bins = np.linspace(x_lims[0], x_lims[1], n_bins)

    plt.figure(figsize=figsize)

    if title is not None:
        plt.title(title)

    for i in range(width*height):
        plt.subplot(height, width, i+1)
        plt.hist(samples[:,i].numpy(), bins=bins, density=True)
        if x_lims is not None:
            plt.xlim(x_lims)
        if i == (width*height - width // 2):
            plt.xlabel(x_label)
