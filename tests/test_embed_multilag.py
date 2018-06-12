import getpass

username = getpass.getuser()

import sys

sys.path.append('/Users/{}/Documents/Reference/G/github/sidpy/sidpy'.format(username))

import sidpy

import numpy

import matplotlib.pyplot as plt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Using a regularly spaced mesh with 
# dt = 1:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dt = 1
Tp = 10
tf = 5
dm = 2

x = numpy.arange(0, 4*(Tp + tf), dt)

print("\nUsing dt = {}...".format(dt))

X = sidpy.embed_ts_multilag(x, dt, Tp, tf)
X_pf = sidpy.extract_multilag_from_embed(X, dt, Tp, tf, dm)

print(X)
print("\n")
print(X_pf)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Using a regularly spaced mesh with 
# dt = 0.1:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dt = 0.1
Tp = 10
tf = 5

dm = 2


x = numpy.arange(0, 4*(Tp + tf), dt)

X = sidpy.embed_ts_multilag(x, dt, Tp, tf)
X_pf = sidpy.extract_multilag_from_embed(X, dt, Tp, tf, dm)

print(X)
print("\n")
print(X_pf)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Using a realization from the stochastic nanopore
# system.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = numpy.loadtxt('../example-data/data-snanopore/NRK5.txt')

dt = 1.0
Tp = 10
tf = 10

time = numpy.arange(0, len(x)*dt, dt)

X = sidpy.embed_ts_multilag(x, dt, Tp, tf)
TIME = sidpy.embed_ts_multilag(time, dt, Tp, tf)

for dm in [1, 2, 4, 5, 8, 10]:
	plt.figure()
	plt.plot(TIME[0, :], X[0, :])

	X_pf = sidpy.extract_multilag_from_embed(X, dt, Tp, tf, dm)
	TIME_pf = sidpy.extract_multilag_from_embed(TIME, dt, Tp, tf, dm)

	plt.plot(TIME_pf[0, :], X_pf[0, :], '.')

plt.show()