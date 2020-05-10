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
