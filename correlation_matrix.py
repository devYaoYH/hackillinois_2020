import h5py
import os
import sys
import math
import time
import itertools
import matplotlib.pyplot as plt
import numpy as np

from utils import filter_data
from utils import norm_zero
from scipy.stats import pearsonr
from scipy.stats import zscore
# seed random number generator

total_samples = 100000

##################
# TIMER FUNCTION #
##################
def debug_time(msg, init, now):
    print("{} {}ms".format(msg, int(round((now-init)*1000*1000))/1000.0), file=sys.stderr)

cwd = os.getcwd()

#Open the data file
data_files = []
for file in os.listdir("data"):
	if file.endswith(".hdf"):
		data_files.append(cwd + "\\data" + f'\\{file}')

# Just take first sample (Small test)
# data_files = data_files[:10]

files = [h5py.File(f,'r') for f in data_files]
datas = [f['DYNAMIC DATA'] for f in files]
sample_slice = total_samples//len(datas)

#Show all channels available in file
chanIDs = files[0]['DYNAMIC DATA']
channels = sorted(list(chanIDs.keys()))

rand_data_slice_init = [np.random.randint(0,len(d[channels[0]]['MEASURED'])-sample_slice) for d in datas]

num_channels = len(channels)
channel_dat = dict()
for ch in channels:
	# norm_t = time.time()
	print(f"Processing: {ch}")
	compound_data = np.asarray([])
	for i, d in enumerate(datas):
		r_init = rand_data_slice_init[i]
		cur_slice = np.asarray(d[ch]['MEASURED'][r_init:r_init + sample_slice])
		cur_slice = filter_data(cur_slice)
		compound_data = np.append(compound_data,cur_slice)
	# debug_time(f"Concat data",norm_t,time.time())
	# channel_dat[ch] = zscore(list(itertools.chain.from_iterable(compound_data)))
	# channel_dat[ch] = zscore(compound_data)
	channel_dat[ch] = norm_zero(compound_data)
	# channel_dat[ch] = compound_data
	# channel_dat[ch] = zscore(datas[0][ch]['MEASURED'])
	# debug_time(f"Normed channel {ch} | {len(compound_data)}",norm_t,time.time())

# Done with loading data, close file
for f in files:
	f.close()

print(f"Channels: {num_channels}")
# num_channels = min(num_channels,10)
corr_M = [[0 for i in range(num_channels)] for j in range(num_channels)]

nan_channels = []
for i in range(num_channels-1):
	ch_a = channels[i]
	dset1 = channel_dat[ch_a]
	for j in range(i,num_channels):
		ch_b = channels[j]
		dset2 = channel_dat[ch_b]
		# corr, _ = pearsonr(dset1, dset2)
		corr = np.dot(dset1,dset2)/len(dset1)
		if (math.isnan(corr)):
			#print(f"Channels {i} and {j} corr = NaN")
			if (i == j):
				nan_channels.append(ch_a)
				# print(f"Inspect Channel {ch_a}")
		else:
			corr_M[i][j] = corr
			corr_M[j][i] = corr

print(nan_channels)

corr_M = np.square(np.asmatrix(corr_M,dtype=float))
# for i in range(corr_M.shape[0]):
# 	for j in range(corr_M.shape[1]):
# 		if (corr_M[i,j] < 0):
# 			corr_M[i,j] = -corr_M[i,j]**2
# 		else:
# 			corr_M[i,j] = corr_M[i,j]**2
np.save("corr_M2",corr_M)

my_dpi=96
plt.figure(figsize=(2048/my_dpi,2048/my_dpi),dpi=my_dpi,frameon=False)
plt.imshow(corr_M, cmap='cool', interpolation='nearest')
# plt.show()
plt.savefig('corr_matrix_full2.png',dpi=my_dpi*2)