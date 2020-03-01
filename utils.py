import math
import numpy as np
from sklearn import linear_model

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