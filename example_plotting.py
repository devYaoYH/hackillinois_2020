import h5py
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import queue

from scipy.stats import zscore

WIN_SIZE = 25000

cwd = os.getcwd()

#Open the data file
data_files = []
for file in os.listdir("data"):
	if file.endswith(".hdf"):
		data_files.append(cwd + "\\data" + f'\\{file}')

# Small sample test, just the first
data_files = data_files[:1]

files = [h5py.File(f,'r') for f in data_files]
f = files[0]

#Show all channels available in file
chanIDs = f['DYNAMIC DATA']

print("Channels available in this data file")
print(list(chanIDs.keys()))

#List of channels to inspect
nan_channels = ['ch_101', 'ch_104', 'ch_107', 'ch_108', 'ch_112', 'ch_15', 'ch_16', 'ch_17', 'ch_18', 'ch_181', 'ch_185', 'ch_187', 'ch_19', 'ch_199', 'ch_2', 'ch_20', 'ch_200', 'ch_21', 'ch_22', 'ch_23', 'ch_24', 'ch_25', 'ch_26', 'ch_262', 'ch_263', 'ch_264', 'ch_267', 'ch_27', 'ch_28', 'ch_29', 'ch_298', 'ch_299', 'ch_3', 'ch_30', 'ch_301', 'ch_302', 'ch_303', 'ch_304', 'ch_33', 'ch_34', 'ch_4', 'ch_45', 'ch_51', 'ch_6', 'ch_65', 'ch_66', 'ch_67', 'ch_71', 'ch_72', 'ch_73', 'ch_74', 'ch_77', 'ch_78', 'ch_79', 'ch_8', 'ch_80', 'ch_81', 'ch_82', 'ch_83', 'ch_85', 'ch_86', 'ch_89', 'ch_92']
problematic_channels = ['ch_4','ch_6','ch_77','ch_78','ch_79','ch_80','ch_81','ch_83','ch_86','ch_89','ch_298']
eighty_channels = [f'ch_{i}' for i in range(80,90)]

# z-score running filter (Mutates data)
def filter_data(data,w_size=100,sd_thresh=1,influence=0):
	M1 = 0
	M2 = 0
	q = queue.Queue()

	# Initialize running avg moments
	for d in data[:w_size]:
		q.put(d)
	M1 = np.mean(data[:w_size])
	M2 = np.mean(np.square(data[:w_size]))
	var = M2 - M1**2
	std = math.sqrt(var) if var != 0 else 0

	for i in range(w_size):
		if (abs(data[i] - M1) > sd_thresh*std):
			data[i] = None

	last_valid_val = 0
	for d in data[:w_size]:
		if (d is not None):
			last_valid_val = d

	# Start moving the window
	for i in range(w_size,len(data)):
		di_val = data[i]
		if (abs(di_val - M1) > sd_thresh*std):
			data[i] = None
			di_val = influence*di_val + (1-influence)*last_valid_val
		if (data[i] is not None):
			last_valid_val = data[i]
		sub_d = q.get()
		M1 -= sub_d/w_size
		M2 -= sub_d**2/w_size
		M1 += di_val/w_size
		M2 += di_val**2/w_size
		q.put(di_val)
	return data

# Fill in None data entries with linear interpolation from boundaries
def interpolate_data(data):
	init_d = -1
	end_d = -1
	inter_range = [0,0]
	for i in range(len(data)+1):
		if (i < len(data) and math.isnan(data[i])):
			if (inter_range[1] < i):
				inter_range[0] = i
				inter_range[1] = i+1
			elif(inter_range[1] == i):
				inter_range[1] = i+1
		else:
			if (inter_range[0] != inter_range[1]):
				if (init_d == -1):
					init_d = data[i] if i < len(data) else 0
				end_d = data[i] if i < len(data) else init_d
				# Linear interpolate between [init_d,end_d]
				steps = (end_d-init_d)/(inter_range[1]-inter_range[0])
				for k in range(inter_range[0],inter_range[1]):
					data[k] = init_d + steps*(k-inter_range[0])
				inter_range[0] = 0
				inter_range[1] = 0
				init_d = data[i]
			else:
				init_d = data[i] if i < len(data) else init_d
	return data

#Plot a sample dataset
def plot_channel(data,ch,norm=True):
	ChannelName = ch
	ch_dat = np.asarray(data[ChannelName]['MEASURED'])
	ch_dat = filter_data(ch_dat,w_size=WIN_SIZE)
	ch_dat = interpolate_data(ch_dat)
	dset = ch_dat
	skip = False
	if (norm):
		std = np.std(dset)
		dset -= np.mean(dset)
		if (std != 0):
			dset /= std
		else:
			skip = True
	if (not skip):
		print(dset[:WIN_SIZE])
		print(f"Num data points: {len(dset)} | Mean: {np.mean(dset)} | Std: {np.std(dset)}")
		plt.plot(dset) # plotting by columns
		plt.title("Value of " + ChannelName)
		plt.xlabel("Datapoint #")
		plt.ylabel("Value")
		plt.show()
	else:
		print(f"{ChannelName} is flat")

test_channels = ['ch_18','ch_84','ch_86','ch_89']
for ch in test_channels:
	plot_channel(chanIDs,ch,norm=True)

#Close the file
f.close()