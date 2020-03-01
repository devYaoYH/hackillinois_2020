import os
import h5py
import math
import numpy as np
import statsmodels.api as sm

from sklearn import linear_model

# Sensor class
class Channel(object):
	def __init__(self,ch,num_channels):
		self.ch = ch
		self.lin_reg = None
		self.covariates = []
		self.corr_mask = np.zeros((num_channels,1))	# Selectively drop non-correlated params
	def load_mask(self,mask):
		self.corr_mask = mask
	def train(self,X,Y,labels,epoch=100):
		self.covariates = []
		X = X.transpose()
		self.lin_reg = LIN_reg(X,Y)
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
		X = X.transpose()
		if (self.lin_reg is None):
			return None
		Y = [self.lin_reg.predict(x) for x in X]
		return Y
	def load(self,lin_reg_sav):
		self.lin_reg = lin_reg_sav

# z-score running filter (Mutates data)
def filter_data(data,w_size=250,sd_thresh=3,influence=0.01):
	M1 = np.mean(data[:w_size])
	vals = np.asarray([M1 for i in range(w_size)])
	cur_idx = 0

	# Initialize running avg moments
	var = np.mean(np.square(vals - M1))
	std = math.sqrt(var) if var != 0 else 0

	for i in range(w_size):
		di_val = data[i]
		if (abs(data[i] - M1) > sd_thresh*std):
			di_val = influence*di_val + (1-influence)*M1
			# data[i] = di_val
			data[i] = None

	# Start moving the window
	for i in range(w_size,len(data)):
		cur_idx = i % w_size
		di_val = data[i]
		if (abs(di_val - M1) > sd_thresh*std):
			di_val = influence*di_val + (1-influence)*vals[(cur_idx+w_size-1)%w_size]
			# data[i] = di_val
			data[i] = None
		vals[cur_idx] = di_val
		M1 = np.mean(vals)
		var = np.mean(np.square(vals - M1))
		std = math.sqrt(var) if var != 0 else 0
	return interpolate_data(data)

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
				init_d = data[i] if i < len(data) else 0
			else:
				init_d = data[i] if i < len(data) else init_d
	return data

def nan_helper(data):
	return np.isnan(data), lambda z: z.nonzero()[0]

def interpolate(data):
	nans, x = nan_helper(data)
	if (len(nans) == 0):
		return data
	data[nans] = np.interp(x(nans), x(~nans), data[~nans])
	return data

def norm_zero(data):
	std = np.std(data)
	data -= np.mean(data)
	if (std != 0):
		return data/std
	else:
		return data

def SGD_log(X,Y,iters=1000):
	print(X.shape)
	print(Y.shape)
	clf = linear_model.SGDClassifier(loss='log')
	clf.fit(X,Y)
	return clf

def LIN_reg(X,Y):
	reg = linear_model.LinearRegression()
	reg.fit(X,Y)
	return reg

def draw_sample(data_path="data",total_samples=10000,filter=filter_data):
	cwd = os.getcwd()
	#Open the data file
	data_files = []
	for file in os.listdir(data_path):
		if file.endswith(".hdf"):
			data_files.append(cwd + f"\\{data_path}\\{file}")

	files = [h5py.File(f,'r') for f in data_files]
	datas = [f['DYNAMIC DATA'] for f in files]
	sample_slice = total_samples//len(datas)
	channels = list(datas[0].keys())

	print([len(d[channels[0]]['MEASURED']) for d in datas])
	print([(0,max(1,len(d[channels[0]]['MEASURED'])-sample_slice)) for d in datas])
	rand_data_slice_init = [np.random.randint(0,max(1,len(d[channels[0]]['MEASURED'])-sample_slice)) for d in datas]

	num_channels = len(channels)
	channel_dat = dict()
	for ch in channels:
		# norm_t = time.time()
		print(f"Processing: {ch}")
		compound_data = np.asarray([])
		for i, d in enumerate(datas):
			r_init = rand_data_slice_init[i]
			cur_slice = np.asarray(d[ch]['MEASURED'][r_init:r_init + sample_slice])
			cur_slice = filter(cur_slice)
			compound_data = np.append(compound_data,cur_slice)
		channel_dat[ch] = norm_zero(compound_data)
	return channels,channel_dat