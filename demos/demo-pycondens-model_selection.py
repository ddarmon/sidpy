#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# A more efficient computation of the predictive
# density, taking advantage of the time series
# structure of the data.
# 
# See notebook pages from 210916 for more details.
# 
# 	DMD, 210916
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from sklearn.neighbors.kde import KernelDensity
from scipy.integrate import quad
import numpy
import scipy
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_distances

import sys

sys.path.append('../sidpy')

import pycondens

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Load the data.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_type = 'nano'

x = numpy.loadtxt('/Users/complexity/Documents/Reference/G/github/sidpy/example-data/data-snanopore/NRK5.txt')
x = x[::2][:1000]

# data_type = 'rr'
# x = numpy.loadtxt('/Users/complexity/Dropbox (Personal)/Reference/T/tirp/2018/hrv-mert/rr-intervals/pid_1001-rm_2017-09-26_17-19-39.dat')[:, 0]
# x = x[~numpy.isnan(x)]
# # x = x[:100]

print 'Remember to *untruncate* the input to spenra.'

p_max = 10

p_opt, nlls, h_raw, active_set_return = pycondens.choose_model_order_nlpl_kde(x, p_max, save_name = data_type, output_verbose = True)