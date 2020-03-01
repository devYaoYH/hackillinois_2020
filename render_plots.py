import h5py
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

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
train1 = "COOLCAT_20100815_055813_69_20100815_055813_695"
fname = "COOLCAT_20110805_194118_39_20110805_194118_391"
test_file = cwd + "\\" + f"train\\{fname}.hdf"

file = h5py.File(test_file,'r')
test_file = file['DYNAMIC DATA']

model = pickle.load(open('normed_mixed_model2.mdl', 'rb'))
m_ch = model['channels']

#Plot a sample dataset
def plot_channel_overlap(channels,g_type="test"):
	_, data_sample,_ = draw_sample(data_path="data",total_samples=1000)
	for ch in channels:
		if (ch not in data_file or ch not in test_file):
			print(f"Channel {ch} does not exit accross datasets")
			continue

		print(f"Plotting Testing Channel {ch}")
		data_dat = data_sample[ch]
		test_dat = filter_data(np.asarray(test_file[ch]['MEASURED'][157340:158340]))

		# plot other dependent graphs too?
		my_dpi = 96
		plt.figure(figsize=(960/my_dpi,960/my_dpi),dpi=my_dpi,frameon=False)
		channel_coef = norm_zero(model['data'][ch].coef_)
		for i in range(len(m_ch)):
			if (abs(channel_coef[i]) > 0.1):
				print(f"Channel {m_ch[i]} is contributive")
				plt.plot(data_sample[m_ch[i]],alpha=0.1)
		plt.plot(data_dat)
		plt.plot(test_dat)
		plt.title("Value of " + ch)
		plt.xlabel("Datapoint #")
		plt.ylabel("Value")
		# plt.show()
		plt.savefig(f'{g_type}_{ch}.png')

# plot_channel_overlap('ch_74')
# plot_channel_overlap('ch_110')

anomalous_channels = ['ch_1', 'ch_109', 'ch_111', 'ch_13', 'ch_14', 'ch_200', 'ch_257', 'ch_266', 'ch_32', 'ch_36', 'ch_40', 'ch_41', 'ch_42', 'ch_48', 'ch_50', 'ch_53', 'ch_59', 'ch_61', 'ch_62', 'ch_68', 'ch_74', 'ch_88', 'ch_89', 'ch_90', 'ch_93']

# Initial unnormed_model.mdl
# anomalous_channels = ['ch_110', 'ch_14', 'ch_32', 'ch_36', 'ch_37', 'ch_43', 'ch_48', 'ch_49', 'ch_53', 'ch_54', 'ch_56', 'ch_58', 'ch_59', 'ch_61', 'ch_90', 'ch_95', 'ch_99']

# Start first 1000 datapoints on mixed_model
# anomalous_channels = ['ch_103', 'ch_111', 'ch_113', 'ch_119', 'ch_226', 'ch_306', 'ch_36', 'ch_43', 'ch_48', 'ch_50', 'ch_52', 'ch_53', 'ch_55', 'ch_56', 'ch_57', 'ch_58', 'ch_61', 'ch_68', 'ch_69', 'ch_75', 'ch_87', 'ch_88', 'ch_9', 'ch_91', 'ch_95', 'ch_96', 'ch_97']

# anomalous_channels = ['ch_103', 'ch_110', 'ch_111', 'ch_226', 'ch_306', 'ch_36', 'ch_37', 'ch_39', 'ch_43', 'ch_44', 'ch_48', 'ch_50', 'ch_52', 'ch_55', 'ch_56', 'ch_60', 'ch_61', 'ch_62', 'ch_64', 'ch_68', 'ch_9', 'ch_91', 'ch_95', 'ch_96', 'ch_98', 'ch_99']

# normal_channels = [r for r in random.sample(list(test_file.keys()), 10) if r not in anomalous_channels]

# for ch in normal_channels:
	# plot_channel_overlap(ch)

# redundant_channels = ['ch_1', 'ch_10', 'ch_100', 'ch_101', 'ch_102', 'ch_103', 'ch_104', 'ch_106', 'ch_107', 'ch_108', 'ch_11', 'ch_110', 'ch_111', 'ch_113', 'ch_114', 'ch_115', 'ch_116', 'ch_117', 'ch_118', 'ch_119', 'ch_12', 'ch_120', 'ch_13', 'ch_132', 'ch_133', 'ch_134', 'ch_14', 'ch_141', 'ch_15', 'ch_16', 'ch_17', 'ch_178', 'ch_18', 'ch_181', 'ch_184', 'ch_185', 'ch_186', 'ch_187', 'ch_19', 'ch_198', 'ch_199', 'ch_2', 'ch_20', 'ch_22', 'ch_23', 'ch_257', 'ch_258', 'ch_261', 'ch_262', 'ch_263', 'ch_267', 'ch_3', 'ch_302', 'ch_303', 'ch_305', 'ch_307', 'ch_308', 'ch_309', 'ch_32', 'ch_33', 'ch_34', 'ch_36', 'ch_37', 'ch_38', 'ch_382', 'ch_383', 'ch_386', 'ch_387', 'ch_39', 'ch_4', 'ch_40', 'ch_41', 'ch_42', 'ch_43', 'ch_44', 'ch_45', 'ch_46', 'ch_48', 'ch_50', 'ch_51', 'ch_53', 'ch_54', 'ch_55', 'ch_57', 'ch_59', 'ch_6', 'ch_61', 'ch_62', 'ch_64', 'ch_66', 'ch_67', 'ch_68', 'ch_71', 'ch_72', 'ch_73', 'ch_74', 'ch_76', 'ch_77', 'ch_78', 'ch_8', 'ch_81', 'ch_82', 'ch_84', 'ch_85', 'ch_86', 'ch_87', 'ch_88', 'ch_89', 'ch_9', 'ch_93', 'ch_94', 'ch_96', 'ch_97', 'ch_98']

plot_channel_overlap(anomalous_channels,g_type="img\\anomalous")