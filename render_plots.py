import h5py
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random

from utils import *

cwd = os.getcwd()

#Open the data file
folder = "data"
data_files = []
for file in os.listdir(folder):
	if file.endswith(".hdf"):
		data_files.append(cwd + f'\\{folder}\\{file}')

# Small sample test, just the first
# data_files = data_files[:1]

files = [h5py.File(f,'r') for f in data_files]
data_file = random.choice(files)['DYNAMIC DATA']

#Open the data file
test_file = cwd + "\\" + "train\\COOLCAT_20100815_055813_69_20100815_055813_695.hdf"

file = h5py.File(test_file,'r')
test_file = file['DYNAMIC DATA']

#Plot a sample dataset
def plot_channel_overlap(ch):
	if (ch not in data_file or ch not in test_file):
		print(f"Channel {ch} does not exit accross datasets")
		return
	data_dat = np.asarray(data_file[ch]['MEASURED'])
	test_dat = np.asarray(test_file[ch]['MEASURED'])

	plt.plot(data_dat,alpha=0.3)
	plt.plot(test_dat)
	plt.title("Value of " + ch)
	plt.xlabel("Datapoint #")
	plt.ylabel("Value")
	plt.show()

plot_channel_overlap('ch_74')

anomalous_channels = []

# Initial unnormed_model.mdl
# anomalous_channels = ['ch_110', 'ch_14', 'ch_32', 'ch_36', 'ch_37', 'ch_43', 'ch_48', 'ch_49', 'ch_53', 'ch_54', 'ch_56', 'ch_58', 'ch_59', 'ch_61', 'ch_90', 'ch_95', 'ch_99']

# Start first 1000 datapoints on mixed_model
# anomalous_channels = ['ch_103', 'ch_111', 'ch_113', 'ch_119', 'ch_226', 'ch_306', 'ch_36', 'ch_43', 'ch_48', 'ch_50', 'ch_52', 'ch_53', 'ch_55', 'ch_56', 'ch_57', 'ch_58', 'ch_61', 'ch_68', 'ch_69', 'ch_75', 'ch_87', 'ch_88', 'ch_9', 'ch_91', 'ch_95', 'ch_96', 'ch_97']

# anomalous_channels = ['ch_103', 'ch_110', 'ch_111', 'ch_226', 'ch_306', 'ch_36', 'ch_37', 'ch_39', 'ch_43', 'ch_44', 'ch_48', 'ch_50', 'ch_52', 'ch_55', 'ch_56', 'ch_60', 'ch_61', 'ch_62', 'ch_64', 'ch_68', 'ch_9', 'ch_91', 'ch_95', 'ch_96', 'ch_98', 'ch_99']

# normal_channels = [r for r in random.sample(list(test_file.keys()), 10) if r not in anomalous_channels]

# for ch in normal_channels:
	# plot_channel_overlap(ch)

for ch in anomalous_channels:
	plot_channel_overlap(ch)