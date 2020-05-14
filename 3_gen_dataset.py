#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pywt

from info import participants, epochs, NUM_BANDS, SLICE_SHAPE, SLICE_WINDOW, SLICE_STEP, SRC_FREQ, TARGET_FREQ, DT


def scale(_x):
    return (_x - _x.min()) / (_x.max() - _x.min())


if __name__ == '__main__':
    print('Loading Data')
    data = pd.read_feather('data/data-clean.ftr')
    print('OK')
    print('Loading Labels')
    labels = pd.read_csv('data/labels.csv', dtype={'ID': object}).set_index('ID')
    label_col = 'ASD'
    score_col = 'ADOS2'
    print('OK')
    BANDS = np.arange(NUM_BANDS) + 1  # frequencies (1 Hz - 50 Hz)

    # define dict to store output
    dataset = {}

    # wavelet transform properties
    wavelet = 'cmor1.5-1.0'  # complex morlet wavelet (Bandwidth - 1.5 Hz, Center Frequency - 1.0 Hz)
    # wavelet = 'sym9'  # symlet 9 wavelet
    scales = SRC_FREQ / BANDS  # scales corresponding to frequency bands

    # generate values for x, y, and z
    print('Generating X, Y, and Z')
    data = data.set_index('Participant')
    for i, p in enumerate(participants):
        print(f'Participant: {p} - ', flush=True, end='')
        label = labels.loc[p][label_col]
        score = labels.loc[p][score_col]
        dp = data.loc[p].set_index('Epoch')
        p_data = np.zeros((0, *SLICE_SHAPE))  # type: np.ndarray
        for j, e in enumerate(epochs[1:]):
            print(f'{e} ', flush=True, end='')
            de = dp.loc[e].set_index('T').to_numpy()  # type: np.ndarray # shape: (timestep, channel)
            # powers of each channel
            ch_p = []
            for ch in de.T:
                # find wavelet transform coefficients of channel signal
                c, _ = pywt.cwt(data=ch, scales=scales, wavelet=wavelet, sampling_period=DT)  # type: np.ndarray
                # calculate abs square of c to obtain wavelet power spectrum
                ps = np.abs(c) ** 2  # type: np.ndarray
                # truncate p to avoid partial slices
                last_t = len(ch) // SRC_FREQ
                last_t -= (last_t - SLICE_WINDOW) % SLICE_STEP
                timesteps = last_t * SRC_FREQ
                l_trim = (len(ch) - timesteps) // 2
                ps = ps[:, l_trim:l_trim + timesteps]
                # down-scale the power spectrum to target frequency (helps to reduce kernel size later)
                E = SRC_FREQ // TARGET_FREQ
                ps = np.amax(ps.reshape((ps.shape[0], ps.shape[1] // E, E)), axis=-1)
                # append power of channel to array
                ch_p.append(ps.T)  # shape: (timestep, band)

            # stack each power spectrum
            ps = np.stack(ch_p, axis=1)  # shape: (timestep, channel, band)
            # chunk power spectrum into N slices of SLICE_SHAPE
            W = SLICE_WINDOW * TARGET_FREQ
            S = SLICE_STEP * TARGET_FREQ
            N = (len(ps) - W) // S
            ws = [ps[k * S:k * S + W].reshape(SLICE_SHAPE) for k in range(N)]
            # generate training data samples
            ds = np.stack(ws, axis=0)  # shape: (sample, timestep, row, col, band)
            # append data samples to participant data
            p_data = np.append(p_data, ds, axis=0)
        # add participant's data to output
        dataset[f'{p}_x'] = p_data
        dataset[f'{p}_y'] = label
        dataset[f'{p}_z'] = score
        print(p_data.shape)
    print('OK')

    # save dataset
    print('Saving processed data')
    np.savez_compressed('data/data-processed.npz', **dataset)
    print('OK')

    # extract delta, theta, alpha, beta, and gamma frequency bands
    print('Reducing to frequency bands')
    dataset = np.load('data/data-processed.npz')
    band_dataset = {}
    for key in dataset.keys():
        if key[-1] != 'x':
            band_dataset[key] = dataset[key]
            continue
        _all_freq_data = dataset[key]
        delta = _all_freq_data[..., 0:4]  # ( <= 4 Hz)
        theta = _all_freq_data[..., 3:8]  # (4 - 8 Hz)
        alpha = _all_freq_data[..., 7:16]  # (8 - 16 Hz)
        beta = _all_freq_data[..., 15:32]  # (16 - 32 Hz)
        gamma = _all_freq_data[..., 31:]  # ( >= 32 Hz)
        _band_data = np.stack([scale(np.max(x, axis=-1)) for x in [delta, theta, alpha, beta, gamma]], axis=-1)  # type: np.ndarray
        band_dataset[key] = _band_data
    print('OK')

    print('Saving frequency band data')
    np.savez_compressed('data/data-processed-bands.npz', **band_dataset)
    print('OK')
