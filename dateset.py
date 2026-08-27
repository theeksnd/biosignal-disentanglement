"""
create_dataset.py — ECG segment sampling for the EPHNOGRAM dataset.

This is an example code (not original)

Draws random 2.0 s ECG windows (resampled to 2000 points) from good-quality
recordings, together with the next segment (t+1) and previous segment (t-1),
and saves them as .npy triplets. Note: sampling is at the segment level with
random start indices; see the manuscript's evaluation-limitations discussion
regarding train/evaluation independence.
"""

import os
import pickle
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

# ---- config ----
DATA_DIR = './dataset/physionet.org/files/ephnogram/1.0.0/WFDB/'
CSV_PATH = './dataset/physionet.org/files/ephnogram/1.0.0/ECGPCGSpreadsheet.csv'
SAVE_DIR = './EvalData/'
WL = 2.0                 # window length in seconds (2.0 s -> 2000 points @ resample)
SEG_POINTS = 2000        # resampled length per segment
N_PER_RECORDING = 10000  # random windows drawn per recording
SEED = 16

# recordings excluded for signal-quality reasons (noise/saturation/disconnection)
IGNORE = ['ECGPCG00{:02d}'.format(k) for k in [3, 4, 5, 6, 7, 8, 9, 40]]

os.makedirs(SAVE_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

df = pd.read_csv(CSV_PATH)
recording_to_segment = {}
index_counter = 0

for record_name, condition, sid in zip(df['Record Name'], df['ECG Notes'], df['Subject ID']):
    if condition != 'Good' or record_name in IGNORE:
        continue

    record = wfdb.rdrecord(os.path.join(DATA_DIR, record_name))
    signal = record.p_signal.T          # -> (channels, samples); channel 0 = ECG
    fs = record.fs
    win = int(WL * fs)

    for _ in range(N_PER_RECORDING):
        # start index leaves room for t-1, t, and t+1 windows
        index = rng.integers(win, signal.shape[-1] - 2 * win)

        seg_t   = resample(signal[0, index:index + win], SEG_POINTS)
        seg_tp1 = resample(signal[0, index + win:index + 2 * win], SEG_POINTS)
        seg_tm1 = resample(signal[0, index - win:index], SEG_POINTS)

        segments = np.stack([seg_t, seg_tp1, seg_tm1], axis=0)   # (3, 2000)

        filename = os.path.join(SAVE_DIR, 's_{}_{}_{}.npy'.format(record_name, sid, index_counter))
        np.save(filename, segments)

        recording_to_segment.setdefault(record_name, []).append(filename)
        index_counter += 1

    print('{}: {} segments'.format(record_name, N_PER_RECORDING))

with open(os.path.join(SAVE_DIR, 'recording_to_segment.pickle'), 'wb') as handle:
    pickle.dump(recording_to_segment, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done. Total segments:', index_counter)
