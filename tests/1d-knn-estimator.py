import getpass

username = getpass.getuser()

import sys

sys.path.append('../sidpy')

import sidpy

import numpy

import scipy.stats

N = 100000

x = numpy.random.rand(N)

hhat = numpy.mean(sidpy.estimate_ler_insample(x, 0, n_neighbors = 5))
h = numpy.log(1)

print("For uniform: h = {}, hhat = {}, |hhat - h| = {}\n\n".format(h, hhat, numpy.abs(hhat - h)))

x = numpy.random.randn(N)

hhat = numpy.mean(sidpy.estimate_ler_insample(x, 0, n_neighbors = 5))
h = 0.5*numpy.log(2*numpy.pi*numpy.exp(1))

print("For normal: h = {}, hhat = {}, |hhat - h| = {}\n\n".format(h, hhat, numpy.abs(hhat - h)))

x = scipy.stats.expon.rvs(size=N)

hhat = numpy.mean(sidpy.estimate_ler_insample(x, 0, n_neighbors = 5))
h = 1 - numpy.log(1)

print("For exponential: h = {}, hhat = {}, |hhat - h| = {}\n\n".format(h, hhat, numpy.abs(hhat - h)))

x = scipy.stats.cauchy.rvs(size=N)

hhat = numpy.mean(sidpy.estimate_ler_insample(x, 0, n_neighbors = 5))
h = numpy.log(4*numpy.pi*1)

print("For cauchy: h = {}, hhat = {}, |hhat - h| = {}\n\n".format(h, hhat, numpy.abs(hhat - h)))

