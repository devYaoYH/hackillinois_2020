import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from significance import StatsAccum

n = 10000
p_array = [0 for i in range(n)]
rho = 0.9
k = 50

acc = StatsAccum(k)

for i in range(n):
  x = np.random.multivariate_normal([0, 0], [[1, rho],[rho, 1]], k)
  for j in range(k):
      acc.add_point(x[j, 0], x[j, 1])

  p = acc.calc_p(rho)
  p_array[i] = p


plt.xlabel("p-value")
plt.ylabel("frequency")

plt.title(f"$\\rho$ = {rho}, $k$ = {k}")

plt.hist(p_array, bins = 25)
plt.show()