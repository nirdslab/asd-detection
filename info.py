#!/usr/bin/env python3

# List of all participants
participants = [
    '002', '004', '005', '007', '008', '011',
    '012', '013', '014', '015', '016', '017',
    '018', '019', '020', '021', '022']

# List of all epochs
epochs = ['BASELINE', 'START', 'MIDDLE', 'END']

# Sampling frequency (250 Hz)
sampling_freq = 250

# Measurement unit (µV)
measurement_unit = 1e-6

# List of all source columns (some columns may be missing in certain files)
source_cols = [
    'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T9', 'T7', 'C3', 'C4', 'T8', 'T10',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'P10',
    'O1'
]

# List of all target columns (interpolate where data is missing)
target_cols = [
    'F9', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'P8', 'P10'
]

# List of target columns with least interpolation (but distorts spatial relationship)
minimal_target_cols = [
    'F9', 'F7', 'F3', 'F4', 'F8', 'F10',
    'FT7', 'FC5', 'FC1', 'FC2', 'FC6', 'FT8',
    'T9', 'T7', 'C3', 'C4', 'T8', 'T10',
    'TP7', 'CP5', 'CP1', 'CP2', 'CP6', 'TP8',
    'P9', 'P7', 'P3', 'P4', 'P8', 'P10'
]

# =============================
# target matrix
# =============================
# F9  F7  F3  [Fz]  F4  F8  F10
# FT7 FC5 FC1 [--]  FC2 FC6 FT8
# T9  T7  C3  [--]  C4  T8  T10
# TP7 CP5 CP1 [--]  CP2 CP6 TP8
# P9  P7  P3  [Pz]  P4  P8  P10
# =============================

# define constant values
NUM_CH_ROWS = 5  # EEG channel rows
NUM_CH_COLS = 10  # EEG channel columns
SRC_FREQ = sampling_freq  # sampling frequency
TARGET_FREQ = 5  # 5 Hz
DT = 1.0 / SRC_FREQ  # sampling period

# define parameters to extract temporal slices
NUM_BANDS = 50  # number of frequency bands in final result
SLICE_WINDOW = 30  # secs per slice
SLICE_STEP = 15  # secs to step to get next slice
SLICE_SHAPE = (SLICE_WINDOW * TARGET_FREQ, NUM_CH_ROWS, NUM_CH_COLS, NUM_BANDS)
SLICE_SHAPE_BANDS = (*SLICE_SHAPE[:-1], 5)
