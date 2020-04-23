#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pywt

if __name__ == '__main__':
    print('Loading cleaned data')
    data = pd.read_feather('data/data-clean.ftr')
    labels = pd.DataFrame(
        data={
            'Label': [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            'Score': [19.0, 12.0, 5.0, 0.0, 5.0, 11.0, 16.0, 16.0, 0.0, 7.0, 4.0, 0.0, 20.0, 2.0, 9.0, 4.0, 0.0]
        },
        index=['002', '004', '005', '007', '008', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022']
    )
    print('OK')

    # define constant values
    participants = data['Participant'].unique()  # participants
    epochs = data['Epoch'].unique()[1:]  # epochs (ignoring baseline)
    NUM_CHANNELS = 30  # number of EEG channels
    FREQ = 250  # sampling frequency
    DT = 1.0 / FREQ  # sampling period

    # define parameters to extract temporal slices
    NUM_BANDS = 50  # number of frequency bands in final result
    BANDS = np.arange(NUM_BANDS) + 1  # frequencies (1 Hz - 50 Hz)
    SLICE_WINDOW = 20  # secs per slice
    SLICE_STEP = 2  # secs to step to get next slice
    SLICE_SHAPE = (SLICE_WINDOW, NUM_BANDS, NUM_CHANNELS)

    # define x, y, and z
    _x = np.empty(shape=(0, *SLICE_SHAPE))  # sample
    _y = np.empty(shape=(0,))  # label
    _z = np.empty(shape=(0,))  # ADOS-2 score

    # wavelet transform properties
    wavelet = 'mexh'  # mexican hat wavelet
    scales = 250 / BANDS  # scales corresponding to frequency bands

    # generate values for x, y, and z
    print('Generating X, Y, and Z')
    data = data.set_index('Participant')
    for i, p in enumerate(participants):
        print(f'Participant: {p} - ', flush=True, end='')
        label = labels.loc[p]['Label']
        score = labels.loc[p]['Score']
        dp = data.loc[p].set_index('Epoch')
        for j, e in enumerate(epochs):
            print(f'{e} ', flush=True, end='')
            de = dp.loc[e].set_index('T').to_numpy()  # type: np.ndarray # shape: (time, channel)
            # wavelet transform on each channel
            transforms = []
            for channel in np.moveaxis(de, 1, 0):  # shape (channel, time)
                # truncate signal to the nearest 1000
                max_frame = len(channel) - len(channel) % 1000
                c, _ = pywt.cwt(data=channel, scales=scales, wavelet=wavelet, sampling_period=DT)  # type: np.ndarray
                c_reduced = np.amax(c[:, :max_frame].reshape((NUM_BANDS, max_frame // 250, 250)), axis=-1)  # type: np.ndarray
                transforms.append(np.transpose(c_reduced))  # shape: (time, band)
            # wavelet decompositions
            wd = np.stack(transforms, axis=-1)  # shape: (time, band, channel)
            # stack sliding windows
            N = (len(wd) - SLICE_WINDOW) // SLICE_STEP
            # windowed samples
            ws = [np.roll(wd, -k * SLICE_STEP, axis=0)[:SLICE_WINDOW].reshape(SLICE_SHAPE) for k in range(N)]
            ds = np.stack(ws, axis=0)  # shape: (sample, time, band, channel)
            _x = np.append(_x, ds, axis=0)
            _y = np.append(_y, np.full((N,), label), axis=0)
            _z = np.append(_z, np.full((N,), score), axis=0)
        print()
    print('OK')

    # save x, y, and z
    print('Saving x, y, and z')
    np.savez_compressed('data/data-final.npz', x=_x, y=_y, z=_z)
    print('OK')
