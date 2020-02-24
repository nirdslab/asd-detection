#!/usr/bin/env python3

import pandas as pd

if __name__ == '__main__':
    participants = [
        '002', '004', '005', '007', '008', '011',
        '012', '013', '014', '015', '016', '017',
        '018', '019', '020', '021', '022']
    epochs = ['BASELINE', 'START', 'MIDDLE', 'END']

    df = pd.DataFrame()

    print('Started')
    for participant in participants:
        filename = f'data/{participant}_TS.xlsx'
        print(f'Reading {filename}...', end=' ', flush=True)
        df_all = pd.read_excel(filename, sheet_name=None)
        sheets = list(df_all.keys())
        for i in range(len(epochs)):
            epoch = epochs[i]
            df_ = df_all[sheets[i]]
            df_['Participant'] = participant
            df_['Epoch'] = epoch
            df = df.append(df_, ignore_index=True)
        print('OK')
    print('Read Completed. Saving Started')
    # reindex
    df.index = range(len(df.index))
    # rename/reorganize columns
    df = df.rename(columns={' ': 'T'})
    cols = ['Participant', 'Epoch', 'T']
    cols += sorted([x for x in df.columns if x not in cols])
    df = df[cols]
    df.to_feather('data/dataset.ftr')
    print('DONE')
