#!/usr/bin/env python3

import pandas as pd

from info import participants, epochs, source_cols

if __name__ == '__main__':
    # DataFrame to store all recordings
    df = pd.DataFrame(columns=['Participant', 'Epoch', *source_cols])
    # read all files into DataFrame
    print(f'Reading data from {len(participants)} participants...')
    for i, participant in enumerate(participants):
        for epoch in epochs:
            filename = f'data/eeg/{participant}/{participant}_{epoch}.csv'
            print(f'\t{filename}...')
            df_ = pd.read_csv(filename)
            df_['Participant'] = participant
            df_['Epoch'] = epoch
            df = df.append(df_, ignore_index=True)
        print(f'{i + 1} of {len(participants)} completed')
    print('Read Completed')
    # rename/reorganize columns and save to file
    print('Saving Data...')
    dest_filename = 'data/data-original.ftr'
    df.to_feather(dest_filename)
    print('DONE')
