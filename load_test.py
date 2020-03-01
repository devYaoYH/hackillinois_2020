import h5py
import os
import pickle
import numpy as np

from utils import *

model = dict()

channels,channel_dat = draw_sample(data_path="data",total_samples=10000)
model['data'] = dict()
model['channels'] = channels

# Load up previously computed correlation matrix (for what?)
corr_M = np.load("corr_M.npy")
# Construct 0-1 matrix from corr_M given threshold
pos = np.where(corr_M > 0.1)
corr_M = np.zeros(corr_M.shape)
for p in pos:
	corr_M[p] = 1
corr_M = corr_M - np.identity(corr_M.shape[0])

# Initialize sensors
sensors = {ch: Channel(ch,len(channels)) for ch in channels}
redundant_channels = []

# Feed compound dataframe into Regressor to train for prediction
for ch in channels:
	print(f"Fitting with linear regression channel: {ch}")
	# Extract current channel as Y
	Y = np.asarray(channel_dat[ch])
	X = np.asmatrix([channel_dat[c] if c != ch else np.zeros(len(channel_dat[c])) for c in channels])
	well_predicted = sensors[ch].train(X,Y,channels)
	if (well_predicted):
		redundant_channels.append(ch)

print(redundant_channels)

# Verify
for ch in channels:
	print(f"{ch}: {sensors[ch].covariates}")
	model['data'][ch] = sensors[ch].lin_reg

pickle.dump(model,open('unnormed_mixed_model.mdl','wb'))