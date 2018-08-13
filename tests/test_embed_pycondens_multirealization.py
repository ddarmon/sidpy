import numpy

import getpass

username = getpass.getuser()

import sys

sys.path.append('../sidpy')

import pycondens
import sidpy

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_distances

# x = numpy.arange(0, 100)
x = numpy.random.rand(1000)

p_max = 10

block_sizes = [.20*x.shape[0], .50*x.shape[0], 0.30*x.shape[0]]
block_sizes = map(int, block_sizes)
block_limits = [0] + numpy.cumsum(block_sizes).tolist()

x_stacked = []

for block_ind in range(len(block_sizes)):
	x_stacked.append(x[block_limits[block_ind]:block_limits[block_ind + 1]])

De_max = pycondens.stack_distance_matrix(x, p_max, is_multirealization = False, output_verbose = False)

X = sidpy.embed_ts(x_stacked, p_max, is_multirealization = True)

D_final = -0.5*numpy.power(pairwise_distances(X, metric = 'euclidean'), 2)

x = x_stacked

ns = []

for r in range(len(x)):
	ns.append(x[r].shape[0])

De_max = pycondens.stack_distance_matrix(x, p_max, is_multirealization = True)

D_final_v2 = numpy.sum(De_max, 2)

dist_error = numpy.abs(D_final - D_final_v2)

print('The maximum error between the two ways of computing the distances is: {}'.format(numpy.max(dist_error)))

fig, ax = plt.subplots(3, 1, sharex = True, sharey = True)
ax[0].matshow(D_final)
ax[1].matshow(D_final_v2)
ax[2].matshow(dist_error)
plt.show()