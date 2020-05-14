import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")

bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

if __name__ == '__main__':
    data = np.load('data/data-processed-bands.npz')
    samples = data['007_x'][0]
    print(samples.shape)
    # plot over time
    fig, axs = plt.subplots(1, 5)
    fig.suptitle('Power Spectrum of EEG Signal')
    for sample in samples:
        for i in range(5):
            axs[i].set_title(bands[i])
            axs[i].imshow(sample[..., i], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        plt.show()
        plt.pause(0.2)
    print('Completed')
