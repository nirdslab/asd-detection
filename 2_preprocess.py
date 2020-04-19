#!/usr/bin/env python3

import mne
import numpy as np
import pandas as pd

participants = {'002', '004', '005', '007', '008', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022'}
if __name__ == '__main__':
    # read data
    print('Loading model...', sep=' ', flush=True)
    df = pd.read_feather('data/dataset.ftr')
    df['FT7'], df['FT8'], df['TP7'], df['TP8'] = np.nan, np.nan, np.nan, np.nan
    print('OK')

    # index by participant, epoch and time
    print('Updating Index')
    df = df.set_index(['Participant', 'Epoch']).sort_index()
    print('OK')

    # target matrix
    # ===========================
    # F9  F7  F3  [Fz]  F4  F8  F10
    # FT7 FC5 FC1 [--]  FC2 FC6 FT8
    # T9  T7  C3  [--]  C4  T8  T10
    # TP7 CP5 CP1 [--]  CP2 CP6 TP8
    # P9  P7  P3  [Pz]  P4  P8  P10
    # ===========================
    target_cols = [
        'F9', 'F7', 'F3', 'F4', 'F8', 'F10',
        'FT7', 'FC5', 'FC1', 'FC2', 'FC6', 'FT8',
        'T9', 'T7', 'C3', 'C4', 'T8', 'T10',
        'TP7', 'CP5', 'CP1', 'CP2', 'CP6', 'TP8',
        'P9', 'P7', 'P3', 'P4', 'P8', 'P10'
    ]
    df_out = pd.DataFrame(columns=target_cols, index=pd.MultiIndex.from_arrays(arrays=[[], [], []], names=['Participant', 'Epoch', 'T']))

    print('Interpolating missing columns')
    for i in df.index.unique():
        # select data slice
        _in = df.loc[i].set_index('T')  # type: pd.DataFrame
        # define columns
        _cols = _in.columns.to_list()  # type: list
        _bads = _in.columns[_in.isna().any()].tolist()  # type: list
        # interpolate bad columns
        _info = mne.create_info(ch_names=_cols, sfreq=250, ch_types='eeg')  # type: dict
        _info['bads'] = _bads
        data = mne.io.RawArray(data=_in.to_numpy().transpose(), info=_info)
        data.set_montage('standard_1020')
        data.interpolate_bads(reset_bads=True)
        # append to output
        _out = data.to_data_frame().rename(columns={'time': 'T'})  # type: pd.DataFrame
        _out['Participant'] = i[0]
        _out['Epoch'] = i[1]
        _out = _out.set_index(['Participant', 'Epoch', 'T'])
        df_out = df_out.append(_out)
    print('OK')

    print('Saving data')
    df_out.reset_index().to_feather('data/dataset-clean.ftr')
    print('OK')
