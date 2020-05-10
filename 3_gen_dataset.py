#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pywt

from info import participants, epochs, sampling_freq

if __name__ == '__main__':
    print('Loading Data')
    data = pd.read_feather('data/data-clean.ftr')
    print('OK')
    print('Loading Labels')
    labels = pd.read_csv('data/labels.csv', dtype={'ID': object}).set_index('ID')
    label_col = 'ASD'
    score_col = 'ADOS2'
    print('OK')

    # define constant values
    NUM_CH_ROWS = 5  # EEG channel rows
    NUM_CH_COLS = 6  # EEG channel columns
    FREQ = sampling_freq  # sampling frequency
    DT = 1.0 / FREQ  # sampling period

    # define parameters to extract temporal slices
    NUM_BANDS = 50  # number of frequency bands in final result
    BANDS = np.arange(NUM_BANDS) + 1  # frequencies (1 Hz - 50 Hz)
    SLICE_WINDOW = 10  # secs per slice
    SLICE_STEP = 10  # secs to step to get next slice
    K = 2  # slice resolution (in Hz)
    SLICE_SHAPE = (SLICE_WINDOW * K, NUM_CH_ROWS, NUM_CH_COLS, NUM_BANDS)

    # define x, y, and z
    _x = np.empty(shape=(0, *SLICE_SHAPE))  # sample
    _y = np.empty(shape=(0,))  # label
    _z = np.empty(shape=(0,))  # ADOS-2 score

    # wavelet transform properties
    wavelet = 'cmor1.0-1.5'  # complex morlet wavelet (1.0 Hz - 1.5 Hz)
    scales = FREQ / BANDS  # scales corresponding to frequency bands

    # generate values for x, y, and z
    print('Generating X, Y, and Z')
    data = data.set_index('Participant')
    for i, p in enumerate(participants):
        print(f'Participant: {p} - ', flush=True, end='')
        label = labels.loc[p][label_col]
        score = labels.loc[p][score_col]
        dp = data.loc[p].set_index('Epoch')
        for j, e in enumerate(epochs[1:]):
            print(f'{e} ', flush=True, end='')
            de = dp.loc[e].set_index('T').to_numpy()  # type: np.ndarray # shape: (timestep, channel)
            # wavelet transform on each channel
            transforms = []
            for channel in np.transpose(de):
                # truncate signal to the nearest 1000
                max_frame = len(channel) - len(channel) % 1000
                c, _ = pywt.cwt(data=channel, scales=scales, wavelet=wavelet, sampling_period=DT)  # type: np.ndarray
                c_reduced = np.amax(c[:, :max_frame].reshape((NUM_BANDS, max_frame // (FREQ // K), (FREQ // K))), axis=-1)  # type: np.ndarray
                transforms.append(np.transpose(c_reduced))  # shape: (timestep, band)
            # wavelet decompositions
            wd = np.stack(transforms, axis=1)  # shape: (timestep, channel, band)
            # stack sliding windows
            N = (len(wd) - (SLICE_WINDOW * K)) // (SLICE_STEP * K)
            # windowed samples
            ws = [np.roll(wd, -k * (SLICE_STEP * K), axis=0)[:(SLICE_WINDOW * K)].reshape(SLICE_SHAPE) for k in range(N)]
            ds = np.stack(ws, axis=0)  # shape: (sample, timestep, row, col, band)
            _x = np.append(_x, ds, axis=0)
            _y = np.append(_y, np.full((N,), label), axis=0)
            _z = np.append(_z, np.full((N,), score), axis=0)
        print()
    print('OK')

    # converting wavelet coefficients to absolute values
    print('Getting absolute values of wavelet coefficients')
    _x = np.abs(_x).astype(np.float32)  # type: np.ndarray
    print('OK')

    # save x, y, and z
    print('Saving processed data')
    np.savez_compressed('data/data-processed.npz', x=_x, y=_y, z=_z)
    print('OK')

    # extract delta, theta, alpha, beta, and gamma frequency bands
    print('Reducing to frequency bands')
    delta = _x[..., 0:4]  # ( <= 4 Hz)
    theta = _x[..., 3:8]  # (4 - 8 Hz)
    alpha = _x[..., 7:16]  # (8 - 16 Hz)
    beta = _x[..., 15:32]  # (16 - 32 Hz)
    gamma = _x[..., 31:]  # ( >= 32 Hz)
    _xb = np.stack([np.max(x, axis=-1) for x in [delta, theta, alpha, beta, gamma]], axis=-1)  # type: np.ndarray
    print('OK')

    print('Saving frequency band data')
    np.savez_compressed('data/data-processed-bands.npz', x=_xb, y=_y, z=_z)
    print('OK')
