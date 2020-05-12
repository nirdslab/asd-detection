import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = np.load('data/data-processed.npz')
    sample = data['002_x'][0]
    print(sample.shape)
    sample = np.reshape(sample, (sample.shape[0], -1))
    plt.imshow(sample.T)
    plt.show()
