from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from info import participants, SLICE_SHAPE_BANDS

matplotlib.use("TkAgg")

band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

if __name__ == '__main__':
    T, H, W, C = SLICE_SHAPE_BANDS
    data = np.load('data/data-processed-bands.npz')

    # plot configuration
    L = 5
    fig, axs = plt.subplots(L, C, sharex='all', sharey='all', figsize=(6, 5))  # type: Figure, List[List[Axes]]
    fig.subplots_adjust(wspace=0.02, hspace=0.04)
    fig.suptitle(f'({H}x{W}) Power Spectrum of {L} Participants over {T} Timesteps')
    ims = []  # type: List[AxesImage]

    for i in range(L):
        axs[i][0].set_ylabel(participants[i])
        for j in range(C):
            axs[0][j].set_title(band_names[j])
            im = axs[i][j].imshow(np.zeros((H, W)), cmap='gray', vmin=0, vmax=1, animated=True)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            ims.append(im)

    def update_fig(_t):
        for _i, _im in enumerate(ims):
            _im.set_array(data[f'{participants[_i // C]}_x'][0, _t, :, :, _i % C])
            yield _im


    ani = animation.FuncAnimation(fig, update_fig, frames=150, blit=True)
    fig.show()
