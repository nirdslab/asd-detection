#!/usr/bin/env python3

from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from info import SLICE_SHAPE_BANDS, participants

band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma'][::-1]

if __name__ == '__main__':
    T, H, W, R = SLICE_SHAPE_BANDS
    data = np.load('data/data-processed-bands.npz')

    # plot configuration
    step = 2
    C = 5
    for p in participants:
        fig, axs = plt.subplots(R, C, sharex='all', sharey='all', figsize=(14, 6))  # type: Figure, List[List[Axes]]
        fig.suptitle(f'Normalized Power Spectrum - Participant {p}')
        # row
        for r in range(R):
            axs[r][0].set_ylabel(band_names[r])
            # column
            for c in range(C):
                t = c * step
                axs[-1][c].set_xlabel(f'{t * 0.2:.0f}')
                # power spectrum
                ax = axs[r][c]
                im = ax.imshow(data[f'{p}_x'][0, t, :, :, r], cmap='inferno', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.colorbar(im, ax=np.ravel(axs).tolist())
        fig.savefig(f'figures/{p}.png', bbox_inches='tight')
        print(f'{p} - OK')
    print('DONE')
