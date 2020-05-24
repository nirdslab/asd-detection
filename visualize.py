#!/usr/bin/env python3

from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from info import EEG_SHAPE, participants

band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

if __name__ == '__main__':
    T, H, W, R = EEG_SHAPE
    data = np.load('data/data-processed-bands.npz')

    # plot configuration
    step = 5
    C = 6
    for p in participants:
        fig, axs = plt.subplots(R, C, sharex='all', sharey='all', figsize=(12, 4))  # type: Figure, List[List[Axes]]
        fig.suptitle(f'Normalized Power Spectrum - Participant {p}')
        # row
        for r in range(R):
            axs[r][0].set_ylabel(band_names[-r - 1])
            # column
            for c in range(C):
                lo = c * step
                hi = (c + 1) * step
                axs[-1][c].set_xlabel(f'({lo}-{hi})')
                # select data - shape: (N, C, H, W, R)
                d = data[f'{p}_x']
                # get exponent of differential entropy, for true power
                d = np.exp(d)
                # select the chunk with highest dispersion
                n = (np.var(d, axis=(1, 2, 3, 4)) / np.mean(d, axis=(1, 2, 3, 4))).argmax()
                d = d[n]
                # normalize by mean of each frequency bin
                d = d / np.mean(d, axis=(0, 1, 2), keepdims=True)
                # rescale data
                d_min = np.min(d)
                d_max = np.max(d)
                d = (d - d_min) / (d_max - d_min)
                # obtain data within time range
                d = np.amax(d[lo:hi], axis=0)
                # power spectrum
                ax = axs[r][c]
                im = ax.imshow(d[:, :, -r - 1], cmap='inferno', vmin=0, vmax=1, interpolation='spline36')
                ax.set_xticks([])
                ax.set_yticks([])
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.colorbar(im, ax=np.ravel(axs).tolist())
        fig.savefig(f'figures/{p}.png', bbox_inches='tight')
        print(f'{p} - OK')
    print('DONE')
