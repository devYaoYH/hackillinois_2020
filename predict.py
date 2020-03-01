import pickle
from utils import *
from significance import get_test_z
from scipy.stats import pearsonr

model = pickle.load(open('unnormed_model.mdl','rb'))
channels = sorted(list(model.keys()))
num_channels = len(channels)

sensors = {ch: Channel(ch,num_channels) for ch in channels}
for ch in channels:
	sensors[ch].load(model[ch])

# Start series prediction
channels, channel_dat = draw_sample(data_path="validation",total_samples=1000)

# Try 1 channel first
for ch in channels[:1]:
	X = np.asmatrix([channel_dat[c] if c != ch else np.zeros(len(channel_dat[c])) for c in channels])
	predicted = sensors[ch].predict(X)
	Y = channel_dat[ch]
	print(f"Prediction Correlation: {pearsonr(Y,predicted)}")