import getpass

username = getpass.getuser()

import sys

sys.path.append('/Users/{}/Documents/Reference/G/github/sidpy/sidpy'.format(username))

import glob

import sidpy

import numpy

p_max = 10

x = []

for trial_ind in range(10):
	# x.append(numpy.arange(trial_ind, 2*p_max))
	x.append(numpy.random.rand(p_max + 1 + trial_ind))

X = sidpy.embed_ts(x, p_max, is_multirealization = True)

print(x)

print(X)