import numpy

import getpass

username = getpass.getuser()

import sys

sys.path.append('/Users/{}/Documents/Reference/G/github/sidpy/sidpy'.format(username))

from sidpy import *

import matplotlib.pyplot as plt

# x = numpy.loadtxt('/Users/complexity/Documents/Reference/G/github/sidpy/example-data/data-shenon/x.dat', delimiter = ',')[:, 0]
# y = numpy.loadtxt('/Users/complexity/Documents/Reference/G/github/sidpy/example-data/data-shenon/y.dat', delimiter = ',')[:, 0]

x = numpy.loadtxt('/Users/complexity/Documents/Reference/G/github/sidpy/example-data/data-shenon/y.dat', delimiter = ',')[:, 0]
y = numpy.loadtxt('/Users/complexity/Documents/Reference/G/github/sidpy/example-data/data-shenon/x.dat', delimiter = ',')[:, 0]

x = x[:1000]
y = y[:1000]

q_opt, p_opt, mse_opt, mse_by_qp, kstar_by_qp = choose_model_order_io_mse(y, x, q_max = 5, p_fix = None, p_max = None, pow_upperbound = 0.5, nn_package = 'sklearn', is_multirealization = False, announce_stages = False, output_verbose = True)

print 'Chose (q* = {}, p* = {}), giving MSE(q*, p*) = {}...'.format(q_opt, p_opt, mse_opt)

for q in range(mse_by_qp.shape[0]):
	print_string = ''
	for p in range(mse_by_qp.shape[1]):
		print_string += '\t{:.3}'.format(mse_by_qp[q, p])

	print print_string
		
plt.matshow(mse_by_qp)
plt.show()