import pandas as pd

if __name__ == '__main__':
    # read data
    df = pd.read_feather('data/dataset.ftr')
    # index by participant, epoch and time
    df = df.set_index(['Participant', 'Epoch', 'T'])
    # convert electrode readings to matrix form
    print(df.columns)
