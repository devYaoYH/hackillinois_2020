import pickle
from utils import *
from significance import get_test_z
from scipy.stats import pearsonr

model = pickle.load(open('normed_mixed_model.mdl','rb'))
channels = model['channels']
channels_idx = {ch: i for i,ch in enumerate(channels)}
num_channels = len(channels)

sensors = {ch: Channel(ch,num_channels) for ch in channels}
for ch in channels:
	sensors[ch].load(model['data'][ch])

# Start series prediction
def run_trail():
	avail_channels, channel_dat = draw_sample(data_path="validation",total_samples=1000)
	data_length = len(channel_dat[list(channel_dat.keys())[0]])

	num_avail_channels = len(avail_channels)

	for i,ch in enumerate(avail_channels):
		if ch not in channels:
			continue
		if (i%(num_avail_channels//10) == 0):
			print(f"Running predictions for {ch} | ...{i*100/num_avail_channels}\%")
		X = np.asmatrix([channel_dat[c] if c != ch and c in channel_dat else np.zeros(data_length) for c in channels])
		predicted = sensors[ch].predict(X).flatten()
		Y = np.asarray(channel_dat[ch])
		try:
			corr, _ = pearsonr(Y,predicted)
		except:
			corr = float("nan")
		if (not math.isnan(corr)):
			# print(f"{ch} Prediction Correlation: {corr}")
			sensors[ch].correlations.append(corr)

def run_test():
	avail_channels, channel_dat = extract_data(data_file="train\\COOLCAT_20100815_055813_69_20100815_055813_695.hdf", max_len=1000)
	# avail_channels, channel_dat = draw_sample(data_path="train",total_samples=10000)
	data_length = len(channel_dat[list(channel_dat.keys())[0]])

	err_ch = []

	for ch in avail_channels:
		if ch not in channels:
			continue
		X = np.asmatrix([channel_dat[c] if c != ch and c in channel_dat else np.zeros(data_length) for c in channels])
		predicted = sensors[ch].predict(X).flatten()
		Y = np.asarray(channel_dat[ch])
		ch_z_err = get_test_z(Y,predicted,sensors[ch].get_exp_corr())
		if (ch_z_err > 3):
			err_ch.append(ch)
			corr, _ = pearsonr(Y,predicted)
			print(f"{ch} z-Score Error: {ch_z_err} | Absolute corr: {corr} | Expected: {sensors[ch].get_exp_corr()}")
	return err_ch

# Validate
print(f"Running correlation validation")
run_trail()
for ch in channels:
	if (abs(sensors[ch].get_exp_corr()) > 0):
		print(f"Sensor {ch} E[corr] = {sensors[ch].get_exp_corr():.5f}")

# Test
err_ch = run_test()
print(err_ch)