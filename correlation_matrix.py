import h5py
import os
import math
import matplotlib.pyplot as plt
import numpy as np

from visualization.Visualizer import Visualizer
from scipy.stats import pearsonr
from scipy.stats import zscore
# seed random number generator

cwd = os.getcwd()

#Open the data file
# fname = "COOLCAT_20091227_194103_36_20091227_194103_360.hdf"
fname = "COOLCAT_20091222_164214_64_20091222_164214_640.hdf"
filepath = cwd + f'\\{fname}'
f = h5py.File(filepath, 'r')

#Show all channels available in file
chanIDs = f['DYNAMIC DATA']
channels = list(chanIDs.keys())
num_channels = len(channels)
channel_dat = dict()
for ch in channels:
	channel_dat[ch] = zscore(chanIDs[ch]['MEASURED'])

print(f"Channels: {num_channels}")
# num_channels = min(num_channels,10)
corr_M = [[0 for i in range(num_channels)] for j in range(num_channels)]

for i in range(num_channels-1):
	ch_a = channels[i]
	dset1 = channel_dat[ch_a]
	for j in range(i,num_channels):
		ch_b = channels[j]
		dset2 = channel_dat[ch_b]
		corr, _ = pearsonr(dset1, dset2)
		if (math.isnan(corr)):
			#print(f"Channels {i} and {j} corr = NaN")
			continue
		else:
			corr_M[i][j] = corr
			corr_M[j][i] = corr

corr_M = np.square(np.asmatrix(corr_M))
my_dpi=96
plt.figure(figsize=(2048/my_dpi,2048/my_dpi),dpi=my_dpi)
plt.imshow(corr_M, cmap='cool', interpolation='nearest')
plt.show()
plt.savefig('corr_matrix.png',dpi=my_dpi*10)

# visualizer = Visualizer(width=800,height=800)
# corr_M_layer = visualizer.add_layer(corr_M.shape)
# corr_M_layer.update(corr_M)
# visualizer.update()
# visualizer.write_img(fname)
# visualizer.pause()