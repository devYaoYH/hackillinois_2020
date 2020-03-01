import h5py
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import queue

from utils import *
from scipy.stats import zscore

WIN_SIZE = 50

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

#Plot a sample dataset
def plot_channel(data,ch,norm=False):
	ChannelName = ch
	ch_dat = np.asarray(data[ChannelName]['MEASURED'])
	ch_dat = filter_data(ch_dat)
	ch_dat = interpolate_data(ch_dat)
	dset = ch_dat
	ch_dat = np.asarray(data[ChannelName]['MEASURED'])
	skip = False
	if (norm):
		std = np.std(dset)
		dset -= np.mean(dset)
		if (std != 0):
			dset /= std
			c_std = np.std(ch_dat)
			ch_dat -= np.mean(ch_dat)
			if (c_std != 0):
				ch_dat /= c_std
			else:
				skip = True
		else:
			skip = True
	if (not skip):
		print(dset[:WIN_SIZE])
		print(f"Num data points: {len(dset)} | Mean: {np.mean(dset)} | Std: {np.std(dset)}")
		plt.plot(ch_dat,alpha=0.3)
		plt.plot(dset) # plotting by columns
		plt.title("Value of " + ChannelName)
		plt.xlabel("Datapoint #")
		plt.ylabel("Value")
		plt.show()
	else:
		print(f"{ChannelName} is flat")

def plot_channel_raw(data,ch,norm=False):
	ChannelName = ch
	ch_dat = np.asarray(data[ChannelName]['MEASURED'])
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

high_prediction_channels = ['ch_1', 'ch_10', 'ch_102', 'ch_103', 'ch_105', 'ch_106', 'ch_11', 'ch_110', 'ch_111', 'ch_13', 'ch_14', 'ch_177', 'ch_184', 'ch_185', 'ch_187', 'ch_19', 'ch_198', 'ch_266', 'ch_298', 'ch_302', 'ch_303', 'ch_305', 'ch_31', 'ch_32', 'ch_35', 'ch_36', 'ch_37', 'ch_38', 'ch_39', 'ch_4', 'ch_40', 'ch_41', 'ch_42', 'ch_43', 'ch_46', 'ch_47', 'ch_48', 'ch_49', 'ch_50', 'ch_52', 'ch_53', 'ch_54', 'ch_55', 'ch_56', 'ch_57', 'ch_58', 'ch_59', 'ch_6', 'ch_60', 'ch_63', 'ch_65', 'ch_67', 'ch_68', 'ch_7', 'ch_71', 'ch_72', 'ch_73', 'ch_74', 'ch_76', 'ch_77', 'ch_78', 'ch_79', 'ch_8', 'ch_80', 'ch_81', 'ch_82', 'ch_83', 'ch_84', 'ch_86', 'ch_87', 'ch_88', 'ch_89', 'ch_9', 'ch_90', 'ch_91', 'ch_94', 'ch_95', 'ch_97', 'ch_98', 'ch_99']

fail_channels = ['ch_40','ch_74']
test_channels = ['ch_86','ch_89']
inverse_channels = ['ch_33','ch_200']
for ch in high_prediction_channels:
	plot_channel(chanIDs,ch)

#Close the file
f.close()