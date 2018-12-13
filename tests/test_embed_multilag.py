import getpass

username = getpass.getuser()

import sys

sys.path.append('../sidpy')

import sidpy
import load_models

import numpy

import matplotlib.pyplot as plt

import matplotlib.patches
Rectangle = matplotlib.patches.Rectangle

from sklearn import neighbors
from sklearn.metrics import pairwise

import pyflann
import nlopt

import scipy

def loocv_mse(n_neighbors, neighbor_inds, X):
	"""
	loocv_mse computes the mean-squared error
	of a k-nearest predictor using 
	leave-one-crossvalidation, where X is 
	the regression matrix where the first
	p columns correspond to the p past values
	and the (p+1) column corresponds to the future.

	loocv_mse is set up to run with nlopt.

	Parameters
	----------
	n_neighbors : float
			The current candidate number of neighbors to use
			for the k-nearest neighbor predictor.

	neighbor_inds : numpy.array
			The array of the nearest neighbors for an
			evaluation point. Each **row** corresponds
			to an evaluation point, and each **column**
			corresponds to the kth-nearest neighbor.

	X : numpy.array
			The regression matrix. The future, which is
			used for both prediction and evaluation, is
			the right-most column.

	Returns
	-------
	mse : float
			The leave-one-out cross-validated mean-squared
			error of the k-nearest neighbor predictor.
	"""

	n_neighbors = int(numpy.floor(n_neighbors))

	err = numpy.power(X[:, -1] - numpy.mean(X[neighbor_inds[:, :n_neighbors], -1], 1), 2)
	mse = numpy.mean(err)
	
	# print s, n_neighbors, mse

	return mse

def choose_lag_mse(X, pow_upperbound = 0.5, nn_package = 'sklearn', announce_stages = False, output_verbose = True, compute_w_farther_past = False):
	Lp_norm = 2.

	p_max = X.shape[1] - 1

	X_full = X.copy()

	mse_by_p = []
	kstar_by_p = []

	if compute_w_farther_past:
		ps = range(1, p_max + 1)
	else:	
		ps = range(2, p_max + 1)

	for p_use in ps:
		X = X_full[:, (p_max - p_use):]

		n_neighbors = int(numpy.ceil(numpy.power(X.shape[0] - 1, pow_upperbound)))

		n_neighbors_upperbound = n_neighbors

		if announce_stages:
			print('Computing nearest neighbor distances using k_upper = {}...'.format(n_neighbors))

		# Compute the nearest neighbor distances and nearest neighbor indices
		# in the marginal space.

		Z = X[:, :p_use]

		if nn_package == 'pyflann':
			flann = pyflann.FLANN()

			neighbor_inds, distances_marg = flann.nn(Z,Z,n_neighbors + 1);

			neighbor_inds = neighbor_inds[:, 1:]
			distances_marg = distances_marg[:, 1:]

			distances_marg = numpy.sqrt(distances_marg) # Since FLANN returns the *squared* Euclidean distance.
		elif nn_package == 'sklearn':
			knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

			knn_out = knn.fit(Z)

			distances_marg, neighbor_inds = knn_out.kneighbors()
		else:
			assert False, "Please select either 'sklearn' or 'pyflann' for nn_package."

		if announce_stages:
			print('Done computing nearest neighbor distances...')

		if announce_stages:
			print('Tuning nearest neighbor number...')

		opt_out = scipy.optimize.minimize_scalar(loocv_mse, bounds = [1.0, n_neighbors_upperbound], method = 'bounded', args = (neighbor_inds, X))

		if announce_stages:
			print('Done tuning nearest neighbor number...')

		k_opt = int(numpy.floor(opt_out['x']))

		kstar_by_p += [k_opt]

		if output_verbose:
			print('For p = {}, chose k* = {} with MSE(k*) = {}'.format(p_use, k_opt, opt_out['fun']))

		mse_by_p += [opt_out['fun']]

		if n_neighbors_upperbound - k_opt <= 10:
			print("####################################################\n# Warning: For p = {}, Nelder-Mead is choosing k* near k_upper = {}.\n# Increase pow_upperbound.\n####################################################""".format(p_use, n_neighbors_upperbound))

	p_opt = ps[numpy.argmin(mse_by_p)]
	mse_opt = numpy.min(mse_by_p)

	return p_opt, mse_opt, mse_by_p, kstar_by_p


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

X = sidpy.embed_ts_multilag(x, dt, Tp, tf)
X_pf = sidpy.extract_multilag_from_embed(X, dt, Tp, tf, dm)

# print(X)
# print("\n")
# print(X_pf)

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

# print(X)
# print("\n")
# print(X_pf)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Using a realization from the stochastic nanopore
# system.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numpy.random.seed(1)

N = 50000
# N = 5000
# N = 1000

# model_name = 'snanopore'
model_name = 'slorenz'
# model_name = 'srossler'
# model_name = 'shadow_crash'

# model_name = 'slogistic'

# model_name = 'lorenz'
# model_name = 'rossler'

x, p_true, model_type = load_models.load_model_data(model_name = model_name, N = N, ds_by = 1)

dt = 1.0
Tp = 20
tf = 30

time = numpy.arange(0, len(x)*dt, dt)

X = sidpy.embed_ts_multilag(x, dt, Tp, tf)
TIME = sidpy.embed_ts_multilag(time, dt, Tp, tf)

# dms = [1, 2, 4, 5, 8, 10]
# dms = [1, 2, 4, 8]
dms = range(1, 6)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Choose the optimal embedding parameters:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mse_by_dm = numpy.zeros((len(dms), Tp+1))
mse_by_dm = numpy.nan*mse_by_dm

for dm_ind, dm in enumerate(dms):
	print('\n\n')
	print("Using dm = {}".format(dm))
	X_pf = sidpy.extract_multilag_from_embed(X, dt, Tp, tf, dm)

	# pow_upperbound = 0.5
	pow_upperbound = 0.66
	# pow_upperbound = 0.75

	if dm == 1:
		p_opt, mse_opt, mse_by_p, kstar_by_p = choose_lag_mse(X_pf, pow_upperbound = pow_upperbound, nn_package = 'sklearn', announce_stages = False, output_verbose = True, compute_w_farther_past = True)
		mse_by_dm[dm_ind, :len(mse_by_p)] = mse_by_p
	else:
		p_opt, mse_opt, mse_by_p, kstar_by_p = choose_lag_mse(X_pf, pow_upperbound = pow_upperbound, nn_package = 'sklearn', announce_stages = False, output_verbose = True)		
		mse_by_dm[dm_ind, 0] = mse_by_dm[0, 0]
		mse_by_dm[dm_ind, 1:len(mse_by_p)+1] = mse_by_p

mse_argmin = numpy.nanargmin(mse_by_dm)

inds_argmin = numpy.unravel_index(mse_argmin, mse_by_dm.shape)

plt.figure()
plt.matshow(numpy.log10(mse_by_dm))
plt.colorbar()
plt.scatter(inds_argmin[1], inds_argmin[0], color = 'red')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Plot the different delay choices:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_cur = 60

win_halfwidth = 100

fig, ax = plt.subplots(len(dms), figsize = (20/1.5, 20/1.5))

for dm_ind, dm in enumerate(dms):
	ax[dm_ind].plot(TIME[:, 0], X[:, 0])

	X_pf = sidpy.extract_multilag_from_embed(X, dt, Tp, tf, dm)
	TIME_pf = sidpy.extract_multilag_from_embed(TIME, dt, Tp, tf, dm)

	ax[dm_ind].scatter(TIME_pf[t_cur, :-2], X_pf[t_cur, :-2], s = 20, color = 'red')
	ax[dm_ind].scatter(TIME_pf[t_cur, -2], X_pf[t_cur, -2], s = 40, color = 'blue')
	ax[dm_ind].scatter(TIME_pf[t_cur, -1], X_pf[t_cur, -1], s = 40, color = 'green')
	# ax[dm_ind].axvline(TIME_pf[t_cur, -2])
	# ax[dm_ind].axvline(TIME[t_cur, 0])

	rect = Rectangle((TIME[t_cur, 0], ax[dm_ind].get_ylim()[0]), width = Tp, height = numpy.diff(ax[dm_ind].get_ylim()), alpha = 0.25, color = 'red')

	ax[dm_ind].add_patch(rect)

	ax[dm_ind].set_xlim([t_cur - win_halfwidth, t_cur + win_halfwidth])

ax[-1].set_xlabel("Time (au)")

plt.figure(figsize = (15, 5))
plt.plot(x)

plt.show()