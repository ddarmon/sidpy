from scipy.integrate import quad
import numpy
import scipy
import matplotlib.pyplot as plt

from itertools import islice

from sklearn.metrics.pairwise import pairwise_distances

from scipy.stats import norm

import sys

sys.path.append('../sidpy')

import pycondens

# data_type = 'nano'
# x = numpy.loadtxt('/Users/complexity/Documents/Reference/G/github/sidpy/example-data/data-snanopore/NRK5.txt')
# x = x[::2][:1000]

data_type = 'rr'
x = numpy.loadtxt('/Users/complexity/Dropbox (Personal)/Reference/T/tirp/2018/hrv-mert/rr-intervals/pid_1001-rm_2017-09-26_17-19-39.dat')[:, 0]
x = x[~numpy.isnan(x)]
# x = x[:100]

p_opt = 4

h, active_set = pycondens.load_bandwidth_o('bw-saved/' + data_type + '-' + str(p_opt), p_opt)

ser = pycondens.estimate_ser_kde(x, p_opt, h, active_set)

fig, ax = plt.subplots(2, sharex = True)
ax[0].plot(x)
ax[1].plot([numpy.nan]*p_opt + ser.tolist())
plt.show()