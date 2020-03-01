import math
import numpy as np
import matplotlib.pyplot as plt

def sample_mean(x, n):
  return 1.0/n * sum(x[0:n])

def sample_var(x, n, mean = None):
  assert(n >= 2)
  if mean is None:
    mean = sample_mean(x, n)

  
  sample_var = 1.0/(n-1) * sum([(x[i] - mean)**2 for i in range(n)])

  return sample_var

def normalize(x, n, mean = None, var = None):

  if mean is None:
    mean = sample_mean(x, n)

  if var is None:
    var = sample_var(x, n, mean = mean)

  std = var**0.5

  return [(x[i] - mean) / std for i in range(n)]

def get_test_z(real, pred, corr):

  n = min(len(real), len(pred))
  if n < 2:
    return False


  mean_real = sample_mean(real, n)
  mean_pred = sample_mean(pred, n)

  var_real = sample_var(real, n, mean = mean_real)
  var_pred = sample_var(pred, n, mean = mean_pred)

  normed_real = normalize(real, n, mean=mean_real, var=var_real)
  normed_pred = normalize(pred, n, mean=mean_pred, var=var_pred)

  product = [normed_real[i] * normed_pred[i] for i in range(n)]

  test_corr = sample_mean(product, n)
  var_corr = sample_var(product, n, mean = test_corr)
  stat_std = math.sqrt(var_corr / n)


  # if corr is far away from test_corr, reject

  z_score = (test_corr - corr)/stat_std


  return z_score


# n = 10000
# z_array = [0 for i in range(n)]

# for i in range(n):
#   x = np.random.multivariate_normal([0, 0], [[1, 0.5],[0.5, 1]], 1000)
#   z_array[i] = test_significance(x[:, 0], x[:, 1], 0.6)


# plt.hist(z_array, bins = 25)
plt.show()

class RotatingQueue():
  def __init__(self, size):
    self.size = size
    self.buffer = [0 for _ in range(size)]
    self.ptr = 0

  def add_pop(self, x):
    old = self.buffer[self.ptr]
    self.buffer[self.ptr] = x
    self.ptr = (self.ptr + 1) % self.size

    return old

class MeanVarAccum():

  def __init__(self, window_size):
    self.window_size = window_size
    self.sum = 0
    self.sum_squares = 0
    self.data_queue = RotatingQueue(window_size)

  def add_point(self, x):
    old_point = self.data_queue.add_pop(x)

    self.sum = self.sum + x - old_point
    self.sum_squares = self.sum_squares + x**2 - old_point**2

  def get_mean(self):
    return 1.0 * self.sum / self.window_size;

  def get_var(self):
    n = self.window_size
    return 1.0 / n * self.sum_squares - self.get_mean() ** 2

  def get_sample_var(self):
    n = self.window_size
    return 1.0 * n / (n - 1) * self.get_var()

class CorrAccum():

  def __init__(self, window_size):
    self.window_size = window_size
    self.product_accum = MeanVarAccum(window_size)
    self.predict_accum = MeanVarAccum(window_size)
    self.sample_accum = MeanVarAccum(window_size)

  def add_point(self, x):
    self.product_accum.add_point(x)
    self.predict_accum.add_point(x)
    self.sample_accum.add_point(x)

  def get_stats():
    # return test_corr, corr_std
    # this finds E((x - mu_x)(y - mu_y)) = E(xy) - mu_x * mu_y
    expectation_XY = self.product_accum.get_mean()
    mean_X = self.sample_accum.get_mean()
    mean_Y = self.predict_accum.get_mean()

    var_X = self.sample_accum.get_var()
    var_Y = self.predict_accum.get_var()

    covariance = expectation_XY - mean_X * mean_Y

    # this finds E(((x - mu_x)(y - mu_y)/sqrt(varX * varY) - test_corr))^2)
    # = E(((x - mu_x)(y - mu_y))^2)/varX/varY - test_corr^2
    # = 
    test_corr = covariance / math.sqrt(var_X * var_Y)




# acc = MeanVarAccum(10)

# for i in range(20):
#   acc.add_point(i)

# print(sample_mean(range(10), 10))
# print(acc.get_mean())

# print(sample_var(range(10), 10))
# print(acc.get_sample_var()  )












