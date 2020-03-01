import h5py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import *

model = dict()

channels,channel_dat = draw_sample(data_path="data",total_samples=10000)
channel_to_idx = {ch: i for i,ch in enumerate(channels)}
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

params_M = []
# Feed compound dataframe into Regressor to train for prediction
for ch in channels:
	print(f"Fitting with linear regression channel: {ch}")
	# Extract current channel as Y
	Y = np.asarray(channel_dat[ch])
	X = np.asmatrix([channel_dat[c] if c != ch else np.zeros(len(channel_dat[c])) for c in channels])
	r2_fit, params = sensors[ch].train(X,Y,channels)
	if (r2_fit > 0.995):
		redundant_channels.append(ch)
	params = norm_zero(params)
	params[channel_to_idx[ch]] = r2_fit
	params_M.append(params)

print(redundant_channels)

params_M = np.asmatrix(params_M)
np.save("params_M",params_M)
params_M = np.square(params_M)
my_dpi=96
plt.figure(figsize=(2048/my_dpi,2048/my_dpi),dpi=my_dpi,frameon=False)
plt.imshow(params_M, cmap='cool', interpolation='nearest')
# plt.show()
plt.savefig('params_matrix_mixed_model.png',dpi=my_dpi*2)

# Verify
for ch in channels:
	print(f"{ch}: {sensors[ch].covariates}")
	model['data'][ch] = sensors[ch].lin_reg

pickle.dump(model,open('normed_mixed_model2.mdl','wb'))