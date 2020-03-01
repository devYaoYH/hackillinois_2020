import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def sample_mean(x, n):
  return 1.0/n * sum(x[0:n])

def sample_var(x, n, mean = None):
  assert(n >= 2)
  if mean is None:
    mean = sample_mean(x, n)

  
  sample_var = 1.0/(n) * sum([(x[i] - mean)**2 for i in range(n)])

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
  stat_std = math.sqrt(var_corr / (n - 1))

  print(f"slow test_corr {test_corr} stat_std {stat_std}")


  # if corr is far away from test_corr, reject

  z_score = (test_corr - corr)/stat_std


  return z_score


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

class StatsAccum():

  def __init__(self, window_size):
    self.k = window_size
    self.n = 0

    self.x_queue = RotatingQueue(self.k)
    self.y_queue = RotatingQueue(self.k)

    self.x_acc = 0
    self.y_acc = 0
    self.x2_acc = 0
    self.y2_acc = 0
    self.xy_acc = 0
    self.x2y2_acc = 0
    self.x2y_acc = 0
    self.xy2_acc = 0

  def add_point(self, x, y):
    # described in "Finding r_i efficiently"
    x_0 = self.x_queue.add_pop(x)
    y_0 = self.y_queue.add_pop(y)

    self.x_acc += x - x_0
    self.y_acc += y - y_0

    self.x2_acc += x**2 - x_0**2
    self.y2_acc += y**2 - y_0**2

    self.xy_acc += x*y - x_0 * y_0

    self.x2y2_acc += (x*y)**2 - (x_0*y_0)**2

    self.x2y_acc += (x**2) * y - (x_0**2) * y_0
    self.xy2_acc += x * (y ** 2) - x_0 * (y_0 ** 2)

    if self.n < self.k:
      self.n += 1

  def calc_stats(self):
    # described in "Finding r_i efficiently"

    if self.n < self.k:
      return None

    k = 1.0 * self.k

    sigma_x = 1 / k * math.sqrt(k * self.x2_acc - (self.x_acc ** 2))
    sigma_y = 1 / k * math.sqrt(k * self.y2_acc - (self.y_acc ** 2))

    #print(f"sigma_x: {sigma_x} sigma_y: {sigma_y}")

    if sigma_x < 1e-6 or sigma_y < 1e-6:
      return None, None



    Exy = 1 / k * self.xy_acc
    
    x_bar = 1 / k * self.x_acc
    y_bar = 1 / k * self.y_acc

    r_i = 1 / (sigma_x * sigma_y) * (Exy - x_bar * y_bar)

    Ex2y2 = 1 / k * self.x2y2_acc

    Ey2 = 1 / k * self.y2_acc
    Ex2 = 1 / k * self.x2_acc

    Exy2 = 1 / k * self.xy2_acc
    Ex2y = 1 / k * self.x2y_acc

    # split it up
    mult = 1.0 / ((sigma_x ** 2) * (sigma_y ** 2))

    term1 = Ex2y2
    term2 = (x_bar ** 2) * Ey2
    term3 = (y_bar ** 2) * Ex2
    term4 = -2 * x_bar * Exy2
    term5 = -2 * y_bar * Ex2y
    term6 = -3 * (x_bar ** 2) * (y_bar ** 2)
    term7 = 4 * x_bar * y_bar * Exy

    Ea2b2 = mult * (term1 + term2 + term3 + term4 + term5 + term6 + term7)

    sample_var = (Ea2b2 - r_i ** 2) / (k-1)
    sample_std = math.sqrt(sample_var)

    return r_i, sample_std

  def calc_p(self, rho):
    # described in "Detecting Anomalies"

    r_i, sample_std = self.calc_stats()

    abs_z = abs((r_i - rho)/sample_std)
    return 2 * norm.cdf(-1 * abs_z)

  def calc_z(self, rho):
    r_i, sample_std = self.calc_stats()
    z = (r_i - rho)/sample_std

    return z

# acc = StatsAccum(10)

# for i in range(20):
#   acc.add_point(i, i % 3)
#   if (i >= 9):
#     print(acc.calc_stats())
#     get_test_z(acc.x_queue.buffer, acc.y_queue.buffer, 10)
    



# acc = MeanVarAccum(10)

# for i in range(20):
#   acc.add_point(i)

# print(sample_mean(range(10), 10))
# print(acc.get_mean())

# print(sample_var(range(10), 10))
# print(acc.get_sample_var()  )










