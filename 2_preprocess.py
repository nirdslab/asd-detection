#!/usr/bin/env python3

import pandas as pd


def fill_missing_values():
    # edge extension
    df['F9'] = df['F9'].fillna((df['F7']) * 0.75)
    df['F10'] = df['F10'].fillna((df['F8']) * 0.75)
    df['P9'] = df['P9'].fillna((df['P7']) * 0.75)
    df['P10'] = df['P10'].fillna((df['P8']) * 0.75)

    # interpolation (equal contribution)
    df['F4'] = df['F4'].fillna((df['Fz'] + df['F8'] + df['FC2'] + df['FC6']) / 4)
    df['P3'] = df['P3'] = df['P3'] * 0.75 + (df['O1'] * 0.25).fillna(df['P3'] * 0.25)
    del df['O1']
    df['FC5'] = df['FC5'].fillna((df['F7'] + df['F3'] + df['C3'] + df['T7']) / 4)
    df['C4'] = df['C4'].fillna((df['FC2'] + df['FC6'] + df['CP2'] + df['CP6']) / 4)
    df['T8'] = df['T8'].fillna((df['FC6'] + df['C4'] + df['T10']) / 3)
    df['CP6'] = df['CP6'].fillna((df['T8'] + df['C4'] + df['CP2'] + df['P4'] + df['P8']) / 5)
    df['CP1'] = df['CP1'].fillna((df['C3'] + df['P3'] + df['CP5']) / 3)
    df['P8'] = df['P8'].fillna((df['P10'] + df['CP6'] + df['P4']) / 3)
    df['P4'] = df['P4'].fillna((df['CP2'] + df['CP6'] + df['P8']) / 3)
    df['Pz'] = df['Pz'].fillna((df['P4'] + df['P3']) / 2)

    # fill missing electrodes with the average of adjacent electrodes, and weight it by 0.75
    df['FC7'] = ((df['FC5'] + df['T7'] + df['F7']) / 3) * 0.75
    df['FC8'] = ((df['FC6'] + df['T8'] + df['F8']) / 3) * 0.75
    df['CP7'] = ((df['CP5'] + df['T7'] + df['P7']) / 3) * 0.75
    df['CP8'] = ((df['CP6'] + df['T8'] + df['P8']) / 3) * 0.75


def find_missing_value_cols():
    for col in df.columns:
        missing_participants = participants.difference(df[col].dropna().reset_index()["Participant"].unique())
        if len(missing_participants) > 0:
            print(f'col_{col.lower()}={missing_participants}')


participants = {'002', '004', '005', '007', '008', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022'}
if __name__ == '__main__':
    # read data
    print('Loading model...', sep=' ', flush=True)
    df = pd.read_feather('data/dataset.ftr')
    print('OK')

    # index by participant, epoch and time
    print('Updating Index')
    df = df.set_index(['Participant', 'Epoch', 'T'])

    # check columns for missing data
    print('Checking columns for missing data')
    find_missing_value_cols()

    # interpolate missing data with nearby electrode values (pre-processing)
    print('Interpolating missing data with nearby values')

    # edge extension part (75%)
    fill_missing_values()

    # make sure there are no missing data
    print('Making sure there are no missing data')
    assert df.shape == df.dropna().shape

    # the resulting matrix looks like
    # ===========================
    # F9  F7  F3  [Fz]  F4  F8  F10
    # FC7 FC5 FC1 [--]  FC2 FC6 FC8
    # T9  T7  C3  [--]  C4  T8  T10
    # CP7 CP5 CP1 [--]  CP2 CP6 CP8
    # P9  P7  P3  [Pz]  P4  P8  P10
    # ===========================

    print('Setting electrode order, and saving dataset')
    cols = [
        'F9', 'F7', 'F3', 'F4', 'F8', 'F10',
        'FC7', 'FC5', 'FC1', 'FC2', 'FC6', 'FC8',
        'T9', 'T7', 'C3', 'C4', 'T8', 'T10',
        'CP7', 'CP5', 'CP1', 'CP2', 'CP6', 'CP8',
        'P9', 'P7', 'P3', 'P4', 'P8', 'P10'
    ]
    df[cols].reset_index().to_feather('data/dataset-clean.ftr')
