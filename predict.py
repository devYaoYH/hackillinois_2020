import pickle
import matplotlib.pyplot as plt

from utils import *
from significance import get_test_z
from significance import StatsAccum
from scipy.stats import pearsonr

WIN_SIZE = 1000
P_TEST = 0.0000001

model = pickle.load(open('normed_mixed_model3.mdl','rb'))
channel_norms = model['norms']
channels = model['channels']
channels_idx = {ch: i for i,ch in enumerate(channels)}
num_channels = len(channels)

sensors = {ch: Channel(ch,num_channels) for ch in channels}
for ch in channels:
	sensors[ch].load(model['data'][ch])

# Start series prediction
def run_trail():
	avail_channels, channel_dat,_ = draw_sample(data_path="validation",total_samples=1000,norm=channel_norms)

	# Normalize data

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

def is_anomalous(predicted,Y,rho):
	corr_accum = StatsAccum(WIN_SIZE)
	for i in range(min(WIN_SIZE,len(Y))):
		corr_accum.add_point(predicted[i],Y[i])
	p_val = corr_accum.calc_p(rho)
	print(f"p: {p_val}")
	if (p_val < P_TEST):
		return p_val
	for i in range(WIN_SIZE,len(Y)):
		corr_accum.add_point(predicted[i],Y[i])
		p_val = corr_accum.calc_p(rho)
		print(f"p: {p_val}")
		if (p_val < P_TEST):
			return p_val
	return -1

def run_test(target_channels=None,dat_file=None):
	# t_file = "COOLCAT_20100815_055813_69_20100815_055813_695"
	t_file = "COOLCAT_20100612_062257_48_20100612_062257_485"
	avail_channels, channel_dat = extract_data(channel_norms,data_file=f"train\\{t_file}.hdf" if dat_file is None else dat_file, max_len=1000)
	# avail_channels, channel_dat = draw_sample(data_path="train",total_samples=10000)
	data_length = len(channel_dat[list(channel_dat.keys())[0]])

	expt_corr = []
	coef_M = []

	err_ch = []

	for ch in avail_channels if target_channels is None else target_channels:
		if ch not in avail_channels or ch not in channels:
			continue
		X = np.asmatrix([channel_dat[c] if c != ch and c in channel_dat else np.zeros(data_length) for c in channels])
		predicted = sensors[ch].predict(X).flatten()
		Y = np.asarray(channel_dat[ch])
		params = sensors[ch].lin_reg.coef_
		coef_M.append(norm_zero(params))
		corr,_ = pearsonr(Y,predicted)
		expt_corr.append(corr)
		corr_exp = sensors[ch].get_exp_corr()
		print(f"{ch} expected corr: {corr_exp} | Actual: {corr}")
		err_p = is_anomalous(predicted,Y,corr_exp)
		if (err_p >= 0):
			err_ch.append(ch)
			print(f"{ch} is Anomalous with {err_p}\% level of significance")
		# ch_z_err = get_test_z(Y,predicted,sensors[ch].get_exp_corr())
		# if (ch_z_err > 3):
		# 	err_ch.append(ch)
		# 	corr, _ = pearsonr(Y,predicted)
		# 	print(f"{ch} z-Score Error: {ch_z_err} | Absolute corr: {corr} | Expected: {sensors[ch].get_exp_corr()}")
	if (target_channels is None):
		return err_ch,expt_corr,coef_M,avail_channels
	else:
		ret_channels = [ch for ch in target_channels if ch in avail_channels and ch in channels]
		return err_ch,expt_corr,coef_M,ret_channels

# Validate
print(f"Running correlation validation")
run_trail()
# for ch in channels:
# 	if (abs(sensors[ch].get_exp_corr()) > 0):
# 		print(f"Sensor {ch} E[corr] = {sensors[ch].get_exp_corr():.5f}")

# Test
test_data_file = "train\\COOLCAT_20110805_194118_39_20110805_194118_391.hdf"
test_channels = None
# test_channels = [f'ch_{i}' for i in range(70,111)]
# test_channels = ['ch_18','ch_40','ch_74','ch_80','ch_90','ch_110']
err_ch,expt_corr,coef_M,testing_channels = run_test(target_channels=test_channels,dat_file=test_data_file)
print(err_ch)

coef_M = np.square(np.asmatrix(coef_M))

my_dpi=96
plt.figure(figsize=(4096/my_dpi,2048/my_dpi),dpi=my_dpi,frameon=False)
plt.xticks(rotation=90)
plt.bar(testing_channels, expt_corr, width=1)
# plt.show()
plt.savefig('corr_testing.png',dpi=my_dpi*2)