import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import *
from sklearn import linear_model

# Sensor class
class Channel(object):
	def __init__(self,ch,num_channels):
		self.ch = ch
		self.log_reg = None
		self.corr_mask = np.zeros((num_channels,1))	# Selectively drop non-correlated params
	def load_mask(self,mask):
		self.corr_mask = mask
	def train(self,X,Y,labels,epoch=100):
		self.covariates = []
		X = X.transpose()
		self.log_reg = LIN_reg(X,Y)
		X2 = sm.add_constant(X)
		est = sm.OLS(Y,X2).fit()
		r2_fit = est.rsquared
		if (r2_fit > 0.995):	# High amount of fit with rest of parameters
			print(f"Channel: {self.ch}:{r2_fit}")
			for i,p in enumerate(est.params):
				if (abs(p) > 0.05):
					if (abs(p > 1)):
						print(f"High correlation w Channel {labels[i-1]}: {p}")
					self.covariates.append(labels[i-1])
			return True
		return False
	def predict(self,X):
		if (self.log_reg is None):
			return None
		return self.log_reg.predict(X)

total_samples = 10000

cwd = os.getcwd()

#Open the data file
data_files = []
for file in os.listdir("data"):
	if file.endswith(".hdf"):
		data_files.append(cwd + "\\data" + f'\\{file}')

files = [h5py.File(f,'r') for f in data_files]
datas = [f['DYNAMIC DATA'] for f in files]
sample_slice = total_samples//len(datas)
channels = list(datas[0].keys())

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
	channel_dat[ch] = norm_zero(compound_data)

corr_M = np.load("corr_M.npy")

# Construct 0-1 matrix from corr_M given threshold
pos = np.where(corr_M > 0.1)
corr_M = np.zeros(corr_M.shape)
for p in pos:
	corr_M[p] = 1
corr_M = corr_M - np.identity(corr_M.shape[0])

# Build and initialize on first dataframe
sensors = {ch: Channel(ch,len(channels)) for ch in channels}
redundant_channels = []

# Feed each dataframe into Regressor to train for prediction
for data in datas[:1]:
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