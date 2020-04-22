#!/usr/bin/env python3

import numpy as np
import pandas as pd

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
    NUM_ROWS = 5
    NUM_COLS = 6
    NUM_CHANNELS = 1
    FREQ = 250  # sampling frequency

    # define parameters to extract temporal slices
    SLICE_WINDOW_SECS = 10  # secs per slice
    SLICE_STEP_SECS = 2  # secs to step to get next slice

    SLICE_WINDOW = FREQ * SLICE_WINDOW_SECS  # samples per slice
    SLICE_STEP = FREQ * SLICE_STEP_SECS  # samples to step to get next slice
    SLICE_SHAPE = (SLICE_WINDOW, NUM_ROWS, NUM_COLS, NUM_CHANNELS)

    # define x, y, and z
    _x = np.empty(shape=(0, *SLICE_SHAPE))  # sample
    _y = np.empty(shape=(0,))  # label
    _z = np.empty(shape=(0,))  # ADOS-2 score

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
            de = dp.loc[e].set_index('T').to_numpy()  # type: np.ndarray
            # stack sliding windows
            N = (len(de) - SLICE_WINDOW) // SLICE_STEP
            ds = np.stack([np.roll(de, -k * SLICE_STEP, axis=0)[:SLICE_WINDOW].reshape(SLICE_SHAPE) for k in range(N)], axis=0)
            _x = np.append(_x, ds, axis=0)
            _y = np.append(_y, np.full((N,), label), axis=0)
            _z = np.append(_z, np.full((N,), score), axis=0)
        print()
    print('OK')

    # save x, y, and z
    print('Saving x, y, and z')
    np.savez_compressed('data/data-final.npz', x=_x, y=_y, z=_z)
    print('OK')
