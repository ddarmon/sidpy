import scipy.special
import numpy

from sklearn import neighbors
from sklearn.metrics import pairwise

import pyflann
import nlopt

from jpype import *
import string

import matplotlib.pyplot as plt

gamma = scipy.special.gamma
digamma = scipy.special.digamma

def choose_model_order_nlpl(x, p_max, pow_upperbound = 0.5, marginal_estimation_procedure = 'knn', nn_package = 'sklearn', is_multirealization = False, announce_stages = False, output_verbose = True, suppress_warning = False):
	"""
	choose_model_order_nlpl computes the negative log-predictive likelihood (NLPL)
	of the  data for varying model orders via a kernel nearest neighbor estimator
	of the predictive density, and returns the model order that minimizes the NLPL.

	Parameters
	----------
	x : list or numpy.array
			The time series as a list (if is_multirealization == False)
			or a numpy.array (if is_multirealization == True).
	p_max : int
			The maximum model order to consider.
	power_upperbound : float
			A number in [0, 1] that determines the upper bound
			on the number of nearest neighbors to consider 
			for the kernel nearest neighbor estimator.
	marginal_estimation_procedure : string
			The estimation procedure used to estimate the 
			differential entropy of x. One of {'knn', 'kde'},
			where 'knn' uses the standard k-nearest neighbor
			estimator of the differential entropy, and 'kde'
			uses a kernel density estimator with the bandwidth
			chosen to minimize the negative log-likelihood.
	nn_package : string
			The package used to compute the nearest neighbors,
			one of {'sklearn', 'pyflann'}. sklearn is an exact
			nearest neighbor search, while pyflann is an
			approximate (and non-deterministic) nearest neighbor
			search.
	is_multirealization : boolean
			Is the time series x given as 
			a single long time series, or in a 
			realization-by-realization format 
			where each row corresponds to a single
			realization?
	announce_stages : boolean
			Whether or not to announce the stages of the estimation
			of the NLPL.

	output_verbose : boolean
			Whether or not to output the NLPL per model order.

	Returns
	-------
	p_opt : int
			The model order that minimized the NLPL.
	nlpl_opt : float
			The minimized value NLPL(p_opt).
	nlpl_by_p : list
			The NLPL as a function of the model order.
	er_knn : float
			The estimate of the (total) entropy rate,
			TER(p_opt).
	ler_knn : numpy.array
			The estimate of the local entropy rate at
			the associated values of x.

	"""

	# Check that pow_upperbound \in [0, 1]

	assert pow_upperbound >= 0 and pow_upperbound <= 1, 'pow_upperbound must be a floating point number in [0, 1].'

	# Normalize the data to have sample mean 0 and
	# sample standard deviation 1.

	if isinstance(x, list):
		x_stacked = numpy.hstack(x)

		x = numpy.array(x)

		x_std = x_stacked.std()

		x = (x - x_stacked.mean())/x_std
	else:
		x_std = x.std()

		x = (x - x.mean())/x_std

	Lp_norm = 2.

	# Embed the time series using a maximum 
	# model order of p_max.

	X_full = embed_ts(x, p_max, is_multirealization = is_multirealization)

	# Compute the NLPL as a function of p.

	nlpl_by_p = []

	for p_use in range(1, p_max+1):
		X = X_full[:, (p_max - p_use):]
		X_train = X

		n_neighbors = int(numpy.ceil(numpy.power(X_train.shape[0] - 1, pow_upperbound)))

		n_neighbors_upperbound = n_neighbors

		Z_train = X_train

		if announce_stages:
			print('Computing nearest neighbor distances using k_upper = {}...'.format(n_neighbors))

		# Compute the nearest neighbor distances and nearest neighbor indices
		# using pyflann in the marginal (past) space.

		# Note that flann.nn takes a training and evaluation set, so we skip over the 
		# nearest neighbor, which corresponds to the evaluation point itself.

		Z_train = X_train[:, :p_use]

		if nn_package == 'pyflann':
			flann = pyflann.FLANN()

			neighbor_inds_train, distances_marg_train = flann.nn(Z_train,Z_train,n_neighbors + 1);

			neighbor_inds_train = neighbor_inds_train[:, 1:]
			# Prior to 141217, this was:
			# distances_marg_train = distances_marg_train[1:]
			# which is incorrect. So why didn't it break more of the code?
			# Should be:
			distances_marg_train = distances_marg_train[:, 1:]

			distances_marg_train = numpy.sqrt(distances_marg_train) # Since FLANN returns the *squared* Euclidean distance.
		elif nn_package == 'sklearn':
			knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

			knn_out = knn.fit(Z_train)

			distances_marg_train, neighbor_inds_train = knn_out.kneighbors()
		else:
			assert False, "Please select either 'sklearn' or 'pyflann' for nn_package."

		if p_use == 1:
			if marginal_estimation_procedure == 'knn':
				n_for_marg = 5

				N = distances_marg_train.shape[0]

				lh_memoryless = numpy.log(distances_marg_train[:, n_for_marg-1]) + numpy.log(N) - digamma(n_for_marg) + numpy.log(2)

				lh_memoryless += numpy.log(x_std)
				h_memoryless = numpy.mean(lh_memoryless)

				nlpl_by_p.append(h_memoryless)

				if output_verbose:
					print('For p = 0, with NLPL(k = {}) = {}'.format(n_for_marg, h_memoryless))
			elif marginal_estimation_procedure == 'kde':
				def local_score_data_marginal(q, grad):
					return score_data_marginal(q, D)

				distances = pairwise.pairwise_distances(x[p_max:].reshape(-1, 1), Y=None, metric="euclidean")

				numpy.fill_diagonal(distances, numpy.nan)

				hstart = 1.06*numpy.power(x.shape[0], -1./(5))

				D = -0.5*distances**2

				# Set various parameters of the optimizer from nlopt:

				opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1) # Use Nelder-Mead
				opt.set_min_objective(local_score_data_marginal) # Set the objective function to be minimized
				opt.set_lower_bounds([1e-5]) # Lowerbound for h
				opt.set_upper_bounds([10.]) # Upperbound for h
				opt.set_ftol_rel(0.0001) # Set the stopping criterion for the relative tolerance. Think of this as the number of 'significant digits' in the function minima.

				# Initialize the parameter values with the pilot bandwidth and 
				# half the upper bound of nearest neighbors, and run the
				# optimization.

				x_opt = opt.optimize([hstart])

				hopt = x_opt[0]

				# Compute the differential entropy estimate, being sure to rescale back to
				# the appropriate sample standard deviation.

				er_marginal, ler_marginal = estimate_ler_marginal([hopt], D)

				h_memoryless =  er_marginal + numpy.log(x_std)
				lh_memoryless = ler_marginal + numpy.log(x_std)

				nlpl_by_p.append(h_memoryless)

		if announce_stages:
			print('Done computing nearest neighbor distances...')

		# Compute the distances between the future points, in the same
		# order as neighbor_inds_train, so that Dtrain_sorted is
		# is pre-sorted from nearest past point to furthest past point.

		Dtrain_sorted = numpy.zeros((n_neighbors, X_train.shape[0]), order = 'F')

		if announce_stages:
			print('Computing distances in future space...')

		for ei in range(X_train.shape[0]):
			xi = X_train[ei, -1]

			Dtrain_sorted[:, ei] = numpy.power(xi - X_train[neighbor_inds_train[ei, :], -1], 2)

		Dtrain_sorted = -0.5*Dtrain_sorted

		if announce_stages:
			print('Done computing distances in future space...')

		if announce_stages:
			print('Tuning bandwidth and nearest neighbor index...')

		# Use the Silverman rule of thumb bandwidth as the 
		# initial guess at the bandwidth.

		h = 1.06*numpy.power(Z_train.shape[0], -1./(5))

		# Need to define a local objective function since 
		# nlopt expects the the objective function to only
		# take the parameters and a gradient as arguments,
		# and we need to pass the data.

		def local_score_data(q, grad):
			return score_data(q, Dtrain_sorted)

		# Set various parameters of the optimizer from nlopt:

		opt = nlopt.opt(nlopt.LN_NELDERMEAD, 2) # Use Nelder-Mead for a 2-parameter optimization problem.
		opt.set_min_objective(local_score_data) # Set the objective function to be minimized
		opt.set_lower_bounds([1e-5, 1.]) # Lowerbound for h and k
		opt.set_upper_bounds([10., n_neighbors_upperbound]) # Upperbound for h and k
		opt.set_ftol_rel(0.0000001) # Set the stopping criterion for the relative tolerance. Think of this as the number of 'significant digits' in the function minima.

		# opt.get_ftol_rel() # Check that opt.set_ftol_rel worked.

		# Initialize the parameter values with the pilot bandwidth and 
		# half the upper bound of nearest neighbors, and run the
		# optimization.

		x_opt = opt.optimize([h, n_neighbors_upperbound/2.])

		h_opt = x_opt[0]
		k_opt = int(numpy.ceil(x_opt[1]))

		if not suppress_warning:
			if n_neighbors_upperbound - k_opt <= 10:
				print("####################################################\n# Warning: For p = {}, Nelder-Mead is choosing k* near k_upper = {}.\n# Increase pow_upperbound.\n####################################################""".format(p_use, n_neighbors_upperbound))

		if announce_stages:
			print('Done tuning bandwidth and nearest neighbor index...')

		if announce_stages:
			print('Scoring data (k_upper = {}):'.format(n_neighbors_upperbound))

		# Compute the NLPL at the optimal values of the bandwidth and
		# nearest neighbor number.

		nlpl_insample = score_data(x_opt,Dtrain_sorted)
		nlpl_insample = nlpl_insample + numpy.log(x_std)

		if announce_stages:
			print('Done scoring data (k_upper = {}):'.format(n_neighbors_upperbound))

		if output_verbose:
			print('For p = {}, with NLPL(h* = {}, k* = {}) = {}'.format(p_use, h_opt, k_opt, nlpl_insample))

		nlpl_by_p.append(nlpl_insample)

	p_opt = numpy.argmin(nlpl_by_p)
	nlpl_opt = numpy.min(nlpl_by_p)
	
	if p_opt == 0:
		er_knn = h_memoryless
		ler_knn = lh_memoryless
	else:
		n_neighbors_untuned = 5

		if announce_stages:
			print('Computing k = {} kNN estimator of entropy rate.'.format(n_neighbors_untuned))

		X = embed_ts(x, p_opt, is_multirealization = is_multirealization)

		distances_marg, distances_joint, nn_inds_marg, nn_inds_joint = compute_nearest_neighbors(X, n_neighbors_untuned, Lp_norm = Lp_norm)

		er_knn, ler_knn = estimate_ter(n_neighbors_untuned, distances_marg, distances_joint, p_opt, Lp_norm)

		er_knn += numpy.log(x_std)
		ler_knn += numpy.log(x_std)

		if announce_stages:
			print('Done computing k = {} kNN estimator of entropy rate.'.format(n_neighbors_untuned))

	return p_opt, nlpl_opt, nlpl_by_p, er_knn, ler_knn

def choose_model_order_io_nlpl(y, x, q_max, p_fix = None, p_max = None, pow_upperbound = 0.5, nn_package = 'sklearn', is_multirealization = False, announce_stages = False, output_verbose = True):
	"""
	choose_model_order_io_nlpl computes the negative log-predictive 
	likelihood (NLPL) of a k-nearest neighbor predictor for the output of 
	the input-output system (y, x) where y is the nominal input and x is
	the nominal output.

	Parameters
	----------
	y : list or numpy.array
			The nominal input time series as a list (if is_multirealization == False)
			or a numpy.array (if is_multirealization == True).
	x : list or numpy.array
			The nominal output time series as a list (if is_multirealization == False)
			or a numpy.array (if is_multirealization == True).
	q_max : int
			The maximum model order to consider for the input process.
	p_fix : int
			The value to fix the model order of the output process.
			For the Self-Predictively Optimal (SPO) version of
			transfer entropy, this should be fixed at the optimal
			model order for predicting the future of the output 
			process without knowledge of the input process.
	p_max : int
			The maximum model order to consider for the output process,
			if p_fix == None.
	power_upperbound : float
			A number in [0, 1] that determines the upper bound
			on the number of nearest neighbors to consider 
			for the k-nearest neighbor estimator.
	nn_package : string
			The package used to compute the nearest neighbors,
			one of {'sklearn', 'pyflann'}. sklearn is an exact
			nearest neighbor search, while pyflann is an
			approximate (and non-deterministic) nearest neighbor
			search.
	is_multirealization : boolean
			Is the time series x given as 
			a single long time series, or in a 
			realization-by-realization format 
			where each row corresponds to a single
			realization?
	announce_stages : boolean
			Whether or not to announce the stages of the estimation
			of the MSE.

	output_verbose : boolean
			Whether or not to output the MSE per model order.

	Returns
	-------
	q_opt : int
			The input model order that minimized the MSE.
	p_opt : int
			The output model order that minimized the MSE,
			or p_fix if p_fix != None.
	nlpl_opt : float
			The minimized value NLPL(q_opt,p_opt).
	nlpl_by_qp : list
			The NLPL as a function of the model orders.

	"""

	if p_fix == None and p_max == None:
		p_max = q_max
		ps = list(range(0, p_max + 1))
	elif p_fix != None:
		p_max = p_fix
		ps = [p_fix]
	else:
		ps = list(range(0, p_max + 1))


	qs = list(range(0, q_max + 1))

	# Check that pow_upperbound \in [0, 1]

	assert pow_upperbound >= 0 and pow_upperbound <= 1, 'pow_upperbound must be a floating point number in [0, 1].'

	# Normalize the data to have sample mean 0 and
	# sample standard deviation 1.

	x_std = x.std()

	x = (x - x.mean())/x_std

	y = (y - y.mean(axis = None))/y.std(axis = None)

	Lp_norm = 2.

	# Embed the input and output time series using a maximum 
	# model orders of q_max and p_max, respectively.

	r_max = numpy.max([p_max, q_max])

	Y_full = embed_ts(y, r_max, is_multirealization = is_multirealization)
	X_full = embed_ts(x, r_max, is_multirealization = is_multirealization)

	# Compute the NLPL as a function of q and p.

	nlpl_by_qp = numpy.empty((len(qs), len(ps)))

	nlpl_by_qp.fill(numpy.nan)

	for p_ind in range(len(ps)):
		p_use = ps[p_ind]
		for q_ind in range(len(qs)):
			q_use = qs[q_ind]

			if p_use == 0 and q_use == 0:
				Z_train = X_full[:, -1].reshape(-1, 1)

				n_for_marg = 5

				if nn_package == 'pyflann':
					flann = pyflann.FLANN()

					neighbor_inds_train, distances_marg_train = flann.nn(Z_train,Z_train,n_for_marg + 1);

					neighbor_inds_train = neighbor_inds_train[:, 1:]

					distances_marg_train = distances_marg_train[:, 1:]
					distances_marg_train = numpy.sqrt(distances_marg_train) # Since FLANN returns the *squared* Euclidean distance.
				elif nn_package == 'sklearn':
					knn = neighbors.NearestNeighbors(n_for_marg, algorithm = 'kd_tree', p = Lp_norm)

					knn_out = knn.fit(Z_train)

					distances_marg_train, neighbor_inds_train = knn_out.kneighbors()
				else:
					assert False, "Please select either 'sklearn' or 'pyflann' for nn_package."

				N = distances_marg_train.shape[0]

				lh_memoryless = numpy.log(distances_marg_train[:, n_for_marg-1]) + numpy.log(N) - digamma(n_for_marg) + numpy.log(2)

				lh_memoryless += numpy.log(x_std)
				h_memoryless = numpy.mean(lh_memoryless)

				nlpl_by_qp[q_ind, p_ind] = h_memoryless
			else:
				Y = Y_full[:, (r_max-q_use):-1]
				X = X_full[:, (r_max-p_use):]

				# Consider doing this via accessing a Z_full array 
				# rather than continuously stacking Y and X arrays.
				Z = numpy.concatenate((Y, X), axis = 1)

				X_train = Z

				n_neighbors = int(numpy.ceil(numpy.power(X_train.shape[0] - 1, pow_upperbound)))

				n_neighbors_upperbound = n_neighbors

				Z_train = X_train

				if announce_stages:
					print('Computing nearest neighbor distances using k_upper = {}...'.format(n_neighbors))

				# Compute the nearest neighbor distances and nearest neighbor indices
				# using pyflann in the marginal (past) space.

				# Note that flann.nn takes a training and evaluation set, so we skip over the 
				# nearest neighbor, which corresponds to the evaluation point itself.

				Z_train = X_train[:, :-1]

				if nn_package == 'pyflann':
					flann = pyflann.FLANN()

					neighbor_inds_train, distances_marg_train = flann.nn(Z_train,Z_train,n_neighbors + 1);

					neighbor_inds_train = neighbor_inds_train[:, 1:]
					# Prior to 141217, this was:
					# distances_marg_train = distances_marg_train[1:]
					# which is incorrect. So why didn't it break more of the code?
					# Should be:
					distances_marg_train = distances_marg_train[:, 1:]

					distances_marg_train = numpy.sqrt(distances_marg_train) # Since FLANN returns the *squared* Euclidean distance.
				elif nn_package == 'sklearn':
					knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

					knn_out = knn.fit(Z_train)

					distances_marg_train, neighbor_inds_train = knn_out.kneighbors()
				else:
					assert False, "Please select either 'sklearn' or 'pyflann' for nn_package."

				if announce_stages:
					print('Done computing nearest neighbor distances...')

				# Compute the distances between the future points, in the same
				# order as neighbor_inds_train, so that Dtrain_sorted is
				# is pre-sorted from nearest past point to furthest past point.

				Dtrain_sorted = numpy.zeros((n_neighbors, X_train.shape[0]), order = 'F')

				if announce_stages:
					print('Computing distances in future space...')

				for ei in range(X_train.shape[0]):
					xi = X_train[ei, -1]

					Dtrain_sorted[:, ei] = numpy.power(xi - X_train[neighbor_inds_train[ei, :], -1], 2)

				Dtrain_sorted = -0.5*Dtrain_sorted

				if announce_stages:
					print('Done computing distances in future space...')

				if announce_stages:
					print('Tuning bandwidth and nearest neighbor index...')

				# Use the Silverman rule of thumb bandwidth as the 
				# initial guess at the bandwidth.

				h = 1.06*numpy.power(Z_train.shape[0], -1./(5))

				# Need to define a local objective function since 
				# nlopt expects the the objective function to only
				# take the parameters and a gradient as arguments,
				# and we need to pass the data.

				def local_score_data(q, grad):
					return score_data(q, Dtrain_sorted)

				# Set various parameters of the optimizer from nlopt:

				opt = nlopt.opt(nlopt.LN_NELDERMEAD, 2) # Use Nelder-Mead for a 2-parameter optimization problem.
				opt.set_min_objective(local_score_data) # Set the objective function to be minimized
				opt.set_lower_bounds([1e-5, 1.]) # Lowerbound for h and k
				opt.set_upper_bounds([10., n_neighbors_upperbound]) # Upperbound for h and k
				opt.set_ftol_rel(0.0000001) # Set the stopping criterion for the relative tolerance. Think of this as the number of 'significant digits' in the function minima.

				# opt.get_ftol_rel() # Check that opt.set_ftol_rel worked.

				# Initialize the parameter values with the pilot bandwidth and 
				# half the upper bound of nearest neighbors, and run the
				# optimization.

				x_opt = opt.optimize([h, n_neighbors_upperbound/2.])

				h_opt = x_opt[0]
				k_opt = int(numpy.ceil(x_opt[1]))

				if n_neighbors_upperbound - k_opt <= 10:
					print("####################################################\n# Warning: For p = {}, Nelder-Mead is choosing k* near k_upper = {}.\n# Increase pow_upperbound.\n####################################################""".format(p_use, n_neighbors_upperbound))

				if announce_stages:
					print('Done tuning bandwidth and nearest neighbor index...')

				if announce_stages:
					print('Scoring data (k_upper = {}):'.format(n_neighbors_upperbound))

				# Compute the NLPL at the optimal values of the bandwidth and
				# nearest neighbor number.

				nlpl_insample = score_data(x_opt,Dtrain_sorted)
				nlpl_insample = nlpl_insample + numpy.log(x_std)

				if announce_stages:
					print('Done scoring data (k_upper = {}):'.format(n_neighbors_upperbound))

				if output_verbose:
					print('For (q = {}, p = {}) chose k* = {} with NLPL(k*) = {}'.format(q_use, p_use, k_opt, nlpl_insample))

				nlpl_by_qp[q_ind, p_ind] = nlpl_insample

	q_opt_ind, p_opt_ind = numpy.unravel_index(numpy.nanargmin(nlpl_by_qp, axis = None), nlpl_by_qp.shape)

	nlpl_opt = numpy.min(nlpl_by_qp, axis = None)

	return qs[q_opt_ind], ps[p_opt_ind], nlpl_opt, nlpl_by_qp

def choose_model_order_mse(x, p_max, pow_upperbound = 0.5, nn_package = 'sklearn', is_multirealization = False, announce_stages = False, output_verbose = True):
	"""
	choose_model_order_mse computes the mean-squared error (MSE) of a k-nearest 
	neighbor predictor for the data for varying model orders, and returns the 
	model order that minimizes the MSE.

	Parameters
	----------
	x : list or numpy.array
			The time series as a list (if is_multirealization == False)
			or a numpy.array (if is_multirealization == True).
	p_max : int
			The maximum model order to consider.
	power_upperbound : float
			A number in [0, 1] that determines the upper bound
			on the number of nearest neighbors to consider 
			for the k-nearest neighbor estimator.
	nn_package : string
			The package used to compute the nearest neighbors,
			one of {'sklearn', 'pyflann'}. sklearn is an exact
			nearest neighbor search, while pyflann is an
			approximate (and non-deterministic) nearest neighbor
			search.
	is_multirealization : boolean
			Is the time series x given as 
			a single long time series, or in a 
			realization-by-realization format 
			where each row corresponds to a single
			realization?
	announce_stages : boolean
			Whether or not to announce the stages of the estimation
			of the MSE.

	output_verbose : boolean
			Whether or not to output the MSE per model order.

	Returns
	-------
	p_opt : int
			The model order that minimized the MSE.
	mse_opt : float
			The minimized value NLPL(p_opt).
	mse_by_p : list
			The MSE as a function of the model order.
	kstar_by_p : list
			The tuned values of k to use for the
			k-nearest neighbor predictor.

	"""

	# Check that pow_upperbound \in [0, 1]

	assert pow_upperbound >= 0 and pow_upperbound <= 1, 'pow_upperbound must be a floating point number in [0, 1].'

	Lp_norm = 2.

	# Embed the time series using a maximum 
	# model order of p_max.

	X_full = embed_ts(x, p_max, is_multirealization = is_multirealization)

	# Compute the NLPL as a function of p.

	mse_by_p = []
	kstar_by_p = []

	if isinstance(x, list):
		x_stacked = numpy.hstack(x)

		mse_by_p = [numpy.mean(numpy.power(x_stacked - numpy.mean(x_stacked), 2))]
		kstar_by_p = [x_stacked.shape[0]]
	else:
		mse_by_p = [numpy.mean(numpy.power(x - numpy.mean(x), 2))]
		kstar_by_p = [len(x)]

	for p_use in range(1, p_max+1):
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

	p_opt = numpy.argmin(mse_by_p)
	mse_opt = numpy.min(mse_by_p)

	return p_opt, mse_opt, mse_by_p, kstar_by_p

def choose_model_order_io_mse(y, x, q_max, p_fix = None, p_max = None, pow_upperbound = 0.5, nn_package = 'sklearn', is_multirealization = False, announce_stages = False, output_verbose = True):
	"""
	choose_model_order_io_mse computes the mean-squared error (MSE) of a
	k-nearest neighbor predictor for the output of the input-output
	system (y, x) where y is the nominal input and x is the nominal output.

	Parameters
	----------
	y : list or numpy.array
			The nominal input time series as a list (if is_multirealization == False)
			or a numpy.array (if is_multirealization == True).
	x : list or numpy.array
			The nominal output time series as a list (if is_multirealization == False)
			or a numpy.array (if is_multirealization == True).
	q_max : int
			The maximum model order to consider for the input process.
	p_fix : int
			The value to fix the model order of the output process.
			For the Self-Predictively Optimal (SPO) version of
			transfer entropy, this should be fixed at the optimal
			model order for predicting the future of the output 
			process without knowledge of the input process.
	p_max : int
			The maximum model order to consider for the output process,
			if p_fix == None.
	power_upperbound : float
			A number in [0, 1] that determines the upper bound
			on the number of nearest neighbors to consider 
			for the k-nearest neighbor estimator.
	nn_package : string
			The package used to compute the nearest neighbors,
			one of {'sklearn', 'pyflann'}. sklearn is an exact
			nearest neighbor search, while pyflann is an
			approximate (and non-deterministic) nearest neighbor
			search.
	is_multirealization : boolean
			Is the time series x given as 
			a single long time series, or in a 
			realization-by-realization format 
			where each row corresponds to a single
			realization?
	announce_stages : boolean
			Whether or not to announce the stages of the estimation
			of the MSE.

	output_verbose : boolean
			Whether or not to output the MSE per model order.

	Returns
	-------
	q_opt : int
			The input model order that minimized the MSE.
	p_opt : int
			The output model order that minimized the MSE,
			or p_fix if p_fix != None.
	mse_opt : float
			The minimized value MSE(q_opt,p_opt).
	mse_by_qp : list
			The MSE as a function of the model orders.
	kstar_by_qp : list
			The tuned values of k to use for the
			k-nearest neighbor predictor as a function
			of the model orders.

	"""

	if p_fix == None and p_max == None:
		p_max = q_max
		ps = list(range(0, p_max + 1))
	elif p_fix != None:
		p_max = p_fix
		ps = [p_fix]
	else:
		ps = list(range(0, p_max + 1))


	qs = list(range(0, q_max + 1))

	# Check that pow_upperbound \in [0, 1]

	assert pow_upperbound >= 0 and pow_upperbound <= 1, 'pow_upperbound must be a floating point number in [0, 1].'

	Lp_norm = 2.

	# Embed the input and output time series using a maximum 
	# model orders of q_max and p_max, respectively.

	r_max = numpy.max([p_max, q_max])

	Y_full = embed_ts(y, r_max, is_multirealization = is_multirealization)
	X_full = embed_ts(x, r_max, is_multirealization = is_multirealization)

	# Compute the MSE as a function of q and p.

	mse_by_qp = numpy.empty((len(qs), len(ps)))
	kstar_by_qp = numpy.empty((len(qs), len(ps)))

	mse_by_qp.fill(numpy.nan)
	kstar_by_qp.fill(numpy.nan)

	for p_ind in range(len(ps)):
		p_use = ps[p_ind]
		for q_ind in range(len(qs)):
			q_use = qs[q_ind]

			if p_use == 0 and q_use == 0:
				mse_by_qp[q_ind, p_ind] = numpy.mean(numpy.power(x - numpy.mean(x), 2))
				kstar_by_qp[q_ind, p_ind] = len(x)
			else:
				Y = Y_full[:, (r_max-q_use):-1]
				X = X_full[:, (r_max-p_use):]

				# Consider doing this via accessing a Z_full array 
				# rather than continuously stacking Y and X arrays.
				Z = numpy.concatenate((Y, X), axis = 1)

				n_neighbors = int(numpy.ceil(numpy.power(Z.shape[0] - 1, pow_upperbound)))

				n_neighbors_upperbound = n_neighbors

				if announce_stages:
					print('Computing nearest neighbor distances using k_upper = {}...'.format(n_neighbors))

				# Compute the nearest neighbor distances and nearest neighbor indices
				# in the marginal space.

				Z_past = Z[:, :-1]

				if nn_package == 'pyflann':
					flann = pyflann.FLANN()

					neighbor_inds, distances_marg = flann.nn(Z_past,Z_past,n_neighbors + 1);

					neighbor_inds = neighbor_inds[:, 1:]
					distances_marg = distances_marg[:, 1:]

					distances_marg = numpy.sqrt(distances_marg) # Since FLANN returns the *squared* Euclidean distance.
				elif nn_package == 'sklearn':
					knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

					knn_out = knn.fit(Z_past)

					distances_marg, neighbor_inds = knn_out.kneighbors()
				else:
					assert False, "Please select either 'sklearn' or 'pyflann' for nn_package."

				if announce_stages:
					print('Done computing nearest neighbor distances...')

				if announce_stages:
					print('Tuning nearest neighbor number...')

				opt_out = scipy.optimize.minimize_scalar(loocv_mse, bounds = [1.0, n_neighbors_upperbound], method = 'bounded', args = (neighbor_inds, Z))

				if announce_stages:
					print('Done tuning nearest neighbor number...')

				k_opt = int(numpy.floor(opt_out['x']))

				kstar_by_qp[q_ind, p_ind] = k_opt

				if output_verbose:
					print('For (q = {}, p = {}) chose k* = {} with MSE(k*) = {}'.format(q_use, p_use, k_opt, opt_out['fun']))

				mse_by_qp[q_ind, p_ind] = opt_out['fun']

				if n_neighbors_upperbound - k_opt <= 10:
					print("####################################################\n# Warning: For (q = {}, p = {}), Nelder-Mead is choosing k* near k_upper = {}.\n# Increase pow_upperbound.\n####################################################""".format(q_use, p_use, n_neighbors_upperbound))

	q_opt_ind, p_opt_ind = numpy.unravel_index(numpy.nanargmin(mse_by_qp, axis = None), mse_by_qp.shape)
	mse_opt = numpy.min(mse_by_qp)

	return qs[q_opt_ind], ps[p_opt_ind], mse_opt, mse_by_qp, kstar_by_qp

def choose_model_order_joint_mse(y, x, q_max, p_max, pow_upperbound = 0.5, nn_package = 'sklearn', is_multirealization = False, announce_stages = False, output_verbose = True):
	"""
	choose_model_order_joint_mse chooses the mean-squared error (MSE) optimal
	model order of a k-nearest neighbor predictor for the joint output (y, x)
	of a stochastic process conditional on its past.

	Parameters
	----------
	y : list or numpy.array
			The nominal input time series as a list (if is_multirealization == False)
			or a numpy.array (if is_multirealization == True).
	x : list or numpy.array
			The nominal output time series as a list (if is_multirealization == False)
			or a numpy.array (if is_multirealization == True).
	q_max : int
			The maximum model order to consider for the input process.
	p_max : int
			The maximum model order to consider for the output process.
	power_upperbound : float
			A number in [0, 1] that determines the upper bound
			on the number of nearest neighbors to consider 
			for the k-nearest neighbor estimator.
	nn_package : string
			The package used to compute the nearest neighbors,
			one of {'sklearn', 'pyflann'}. sklearn is an exact
			nearest neighbor search, while pyflann is an
			approximate (and non-deterministic) nearest neighbor
			search.
	is_multirealization : boolean
			Is the time series x given as 
			a single long time series, or in a 
			realization-by-realization format 
			where each row corresponds to a single
			realization?
	announce_stages : boolean
			Whether or not to announce the stages of the estimation
			of the MSE.

	output_verbose : boolean
			Whether or not to output the MSE per model order.

	Returns
	-------
	q_opt : int
			The input model order that minimized the MSE.
	p_opt : int
			The output model order that minimized the MSE.
	mse_opt : float
			The minimized value MSE(q_opt,p_opt).
	mse_by_qp : list
			The MSE as a function of the model orders.
	kstar_by_qp : list
			The tuned values of k to use for the
			k-nearest neighbor predictor as a function
			of the model orders.

	"""

	ps = list(range(0, p_max + 1))
	qs = list(range(0, q_max + 1))

	# Check that pow_upperbound \in [0, 1]

	assert pow_upperbound >= 0 and pow_upperbound <= 1, 'pow_upperbound must be a floating point number in [0, 1].'

	Lp_norm = 2.

	# Embed the input and output time series using a maximum 
	# model orders of q_max and p_max, respectively.

	r_max = numpy.max([p_max, q_max])

	Y_full = embed_ts(y, r_max, is_multirealization = is_multirealization)
	X_full = embed_ts(x, r_max, is_multirealization = is_multirealization)

	# Compute the MSE as a function of q and p.

	mse_by_qp = numpy.empty((len(qs), len(ps)))
	kstar_by_qp = numpy.empty((len(qs), len(ps)))

	mse_by_qp.fill(numpy.nan)
	kstar_by_qp.fill(numpy.nan)

	for p_ind in range(len(ps)):
		p_use = ps[p_ind]
		for q_ind in range(len(qs)):
			q_use = qs[q_ind]

			if p_use == 0 and q_use == 0:
				mse_by_qp[q_ind, p_ind] = numpy.mean(numpy.power(y - numpy.mean(y), 2)) + numpy.mean(numpy.power(x - numpy.mean(x), 2))
				kstar_by_qp[q_ind, p_ind] = len(x)
			else:
				Y = Y_full[:, (r_max-q_use):-1]
				X = X_full[:, (r_max-p_use):-1]

				# Consider doing this via accessing a Z_full array 
				# rather than continuously stacking Y and X arrays.
				Z = numpy.concatenate((Y, X, Y_full[:, -1].reshape(-1, 1), X_full[:, -1].reshape(-1, 1)), axis = 1)

				n_neighbors = int(numpy.ceil(numpy.power(Z.shape[0] - 1, pow_upperbound)))

				n_neighbors_upperbound = n_neighbors

				if announce_stages:
					print('Computing nearest neighbor distances using k_upper = {}...'.format(n_neighbors))

				# Compute the nearest neighbor distances and nearest neighbor indices
				# in the marginal space.

				Z_past = Z[:, :-2]

				if nn_package == 'pyflann':
					flann = pyflann.FLANN()

					neighbor_inds, distances_marg = flann.nn(Z_past,Z_past,n_neighbors + 1);

					neighbor_inds = neighbor_inds[:, 1:]
					distances_marg = distances_marg[:, 1:]

					distances_marg = numpy.sqrt(distances_marg) # Since FLANN returns the *squared* Euclidean distance.
				elif nn_package == 'sklearn':
					knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

					knn_out = knn.fit(Z_past)

					distances_marg, neighbor_inds = knn_out.kneighbors()
				else:
					assert False, "Please select either 'sklearn' or 'pyflann' for nn_package."

				if announce_stages:
					print('Done computing nearest neighbor distances...')

				if announce_stages:
					print('Tuning nearest neighbor number...')

				opt_out = scipy.optimize.minimize_scalar(loocv_mse, bounds = [1.0, n_neighbors_upperbound], method = 'bounded', args = (neighbor_inds, Z))

				if announce_stages:
					print('Done tuning nearest neighbor number...')

				k_opt = int(numpy.floor(opt_out['x']))

				kstar_by_qp[q_ind, p_ind] = k_opt

				if output_verbose:
					print('For (q = {}, p = {}) chose k* = {} with MSE(k*) = {}'.format(q_use, p_use, k_opt, opt_out['fun']))

				mse_by_qp[q_ind, p_ind] = opt_out['fun']

				if n_neighbors_upperbound - k_opt <= 10:
					print("####################################################\n# Warning: For (q = {}, p = {}), Nelder-Mead is choosing k* near k_upper = {}.\n# Increase pow_upperbound.\n####################################################""".format(q_use, p_use, n_neighbors_upperbound))

	q_opt_ind, p_opt_ind = numpy.unravel_index(numpy.nanargmin(mse_by_qp, axis = None), mse_by_qp.shape)
	mse_opt = numpy.min(mse_by_qp)

	return qs[q_opt_ind], ps[p_opt_ind], mse_opt, mse_by_qp, kstar_by_qp

def compute_nearest_neighbors_1d(X, n_neighbors, Lp_norm = 2):
	Z = X

	knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

	knn_out = knn.fit(Z)

	distances, nn_inds = knn_out.kneighbors()

	return distances, nn_inds

def compute_nearest_neighbors(X, n_neighbors, Lp_norm = 2):
	Z = X

	knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

	knn_out = knn.fit(Z)

	distances_joint, nn_inds_joint = knn_out.kneighbors()

	Z = X[:, :-1]

	knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

	knn_out = knn.fit(Z)

	distances_marg, nn_inds_marg = knn_out.kneighbors()

	return distances_marg, distances_joint, nn_inds_marg, nn_inds_joint

def compute_nearest_neighbors_cross(Xfit, Xeval, n_neighbors, Lp_norm = 2):
	knn = neighbors.NearestNeighbors(n_neighbors, algorithm = 'kd_tree', p = Lp_norm)

	knn_out = knn.fit(Xfit)

	distances, neighbor_inds = knn_out.kneighbors(Xeval)

	return distances

def embed_ts(x, p_max, is_multirealization = False):
	"""
	embed_ts transforms a scalar time series x
	into a data matrix where each row t
	corresponds to

		[X_{t}, X_{t + 1}, ..., X_{t + p_max}]

	Thus, a predictor would use the first p_max
	columns of the data matrix to predict the
	(p_max + 1)th column.

	Parameters
	----------
	x : list
			The scalar time series to be embedded.
	p_max : integer
			The maximum model order to consider.
	is_multirealization : boolean
			Is the time series x given as 
			a single long time series, or in a 
			realization-by-realization format 
			where each row corresponds to a single
			realization?

	Returns
	-------
	X : numpy.array
			The (len(x) - p_max) x (p_max + 1) data
			matrix resulting from embedding x.

	"""

	if is_multirealization:
		if isinstance(x, numpy.ndarray): # Check if the realizations are stored in a numpy.array
			n_trials = x.shape[0]
			n = x.shape[1]
			n_per_trial = n - p_max

			X = numpy.empty((n_per_trial*n_trials, p_max+1))

			for trial_ind in range(n_trials):
				x_cur = x[trial_ind, :]
				for p in range(p_max+1):
					X[trial_ind*n_per_trial:(trial_ind + 1)*n_per_trial, p] = x_cur[p:n-(p_max - p)]
		else: # If not, they should be stored in either a list or tuple of arrays.
			n_trials = len(x)

			n_by_trial = numpy.array([trial.shape[0] for trial in x])

			n_per_trial = n_by_trial - p_max

			X = numpy.empty((numpy.sum(n_per_trial), p_max + 1))

			for trial_ind in range(n_trials):
				x_cur = x[trial_ind]

				for p in range(p_max+1):
					X[numpy.sum(n_per_trial[:trial_ind]):numpy.sum(n_per_trial[:trial_ind+1]), p] = x_cur[p:n_by_trial[trial_ind]-(p_max - p)]




	else:
		X = numpy.empty((len(x) - p_max, p_max+1))
		n = len(x)

		for p in range(p_max+1):
			X[:, p] = x[p:n-(p_max - p)]

	return X

def embed_ts_multihorizon(x, p_max, q, is_multirealization = False):
	if is_multirealization:
		N_trials = x.shape[0]
		N = x.shape[1]
		N_per_trial = N - (p_max + q)

		X = numpy.empty((N_per_trial*N_trials, p_max+1))

		for trial_ind in range(N_trials):
			x_cur = x[trial_ind, :]
			for p in range(p_max):
				X[trial_ind*N_per_trial:(trial_ind + 1)*N_per_trial, p] = x_cur[p:N-(q+p_max)+p]

			X[trial_ind*N_per_trial:(trial_ind + 1)*N_per_trial, p_max] = x_cur[(p_max+q):N]

	else:
		N = len(x)

		X = numpy.empty((N - (p_max+q), p_max+1))

		for p in range(p_max):
			X[:, p] = x[p:N-(q+p_max)+p]

		X[:, p_max] = x[(p_max+q):N]

	return X

def score_data(q, D, verbose = False):
	"""
	score_data computes the negative log-predictive
	likelihood (NLPL) of a data set using a kernel
	nearest neighbor estimator of the predictive density.

	score_data is set up to run with nlopt.

	Parameters
	----------
	q : list
			The parameters of the kernel nearest neighbor
			predictive density estimator.

			q[0] = the bandwidth in the future space
			q[1] = the number of nearest neighbors to use

	D : numpy.array
			-0.5 * (squared distance) from an evaluation point
			to an example point, used in the exponenet of the
			kernel for the kernel nearest neighbor estimator.

			D is formatted such that each i column corresponds
			to the -0.5 * (squared distance) from that 
			evaluation point to all of the other future points:
				D[j, i] = -0.5 * (squared distance) from evaluation point i to future point j
			Thus, D ~ (n_neighbors_upperbound) x (len(x) - p_max).

	verbose : boolean
			Whether to output the parameters and NLPL on each call of
			score_data.

	Returns
	-------
	nlpl : float
			The negative log-predictive likelihood of the data using
			the kernel nearest neighbor estimator of the predictive
			density.

	"""
	h = q[0]
	n_neighbors = int(numpy.ceil(q[1]))

	C = 1/(numpy.sqrt(2*numpy.pi)*h)

	K = C*numpy.exp(D/(h**2))

	# Note that K is already pre-sorted:

	fs = numpy.mean(K[:n_neighbors, :], 0)
	log_fs = numpy.log(fs)

	# Without removing infinite entries:
	nlpl = -numpy.mean(log_fs) # Unweighted TER
	# nlpl = -numpy.mean(fs*log_fs) # Weighted TER, a la Tong's paper

	# With removing infinite entries:
	# nlpl = -numpy.mean(log_fs[~numpy.isinf(log_fs)]) # Unweighted TER
	# nlpl = -numpy.mean((fs*log_fs)[~numpy.isinf(log_fs)]) # Weighted TER, a la Tong's paper

	if verbose:
		print(h, q[1], n_neighbors, nlpl)

	return nlpl

def score_data_marginal(q, D, verbose = False):
	"""
	score_data_marginal computes the negative log-
	likelihood of the marginal data using a kernel
	density estimator.

	score_data_marginal is set up to run with nlopt.

	Parameters
	----------
	q : list
			The parameters of the 

			q[0] = the bandwidth in the future space

	D : numpy.array
			-0.5 * (squared distance) from an evaluation point
			to an example point, used in the exponenet of the
			kernel for the kernel density estimator.

	verbose : boolean
			Whether to output the parameters and NLPL on each call of
			score_data_marginal.

	Returns
	-------
	nlpl : float
			The negative log-likelihood of the data using
			the kernel density estimator.

	"""
	h = q[0]

	C = 1/(numpy.sqrt(2*numpy.pi)*h)

	K = C*numpy.exp(D/(h**2))

	fs = numpy.nanmean(K, 0)
	log_fs = numpy.log(fs)

	# Without removing infinite entries:
	nlpl = -numpy.mean(log_fs) # Unweighted TER

	if verbose:
		print(h, nlpl)

	return nlpl

def estimate_ler_marginal(q, D, verbose = False):
	"""
	estimate_ler_marginal computes the local entropy using the
	marginal density via a kernel density estimator.

	Parameters
	----------
	q : list
			The parameters of the 

			q[0] = the bandwidth in the future space

	D : numpy.array
			-0.5 * (squared distance) from an evaluation point
			to an example point, used in the exponenet of the
			kernel for the kernel density estimator.

	verbose : boolean
			Whether to output the parameters and NLPL on each call of
			score_data_marginal.

	Returns
	-------
	er : float
			The average of the local entropies, an estimator for the 
			the total entropy.
	ler : numpy.array
			The local entropy at each of the evaluation points.

	"""
	h = q[0]

	C = 1/(numpy.sqrt(2*numpy.pi)*h)

	K = C*numpy.exp(D/(h**2))

	fs = numpy.nanmean(K, 0)
	log_fs = numpy.log(fs)

	ler = -log_fs

	# Without removing infinite entries:
	er = numpy.mean(ler) # Unweighted TER

	return er, ler

def estimate_ter(n_neighbors, distances_marg, distances_joint, d, Lp_norm):
	"""
	Estimate the total entropy rate given the distances between kth-nearest
	neighbors in the marginal and joint spaces, using the estimator

	\hat{h}[X_{t} | X_{t - d}^{t - 1}] = \hat{h}[X_{t - d}^{t}] - \hat{h}[X_{t - d}^{t - 1}],

	e.g. the conditional entropy equals the joint entropy minus the marginal entropy,
	where the joint and marginal entropies are estimated separately via kth-nearest
	neighbor estimators.


	Parameters
	----------
	n_neighbors : int
			The number of the nearest neighbor to compute the distance
			to, i.e. k for a kth-nearest neighbor estimator.
	distances_marg : numpy.array
			The distances to the k-nearest neighbors of each of
			the evaluation points, in the marginal space.
			Each row corresponds to  an evaluation point, and 
			each column corresponds to a jth-nearest neighbor.
			For example,
				distances_marg[i, j]
			is the distance between the ith evaluation point and
			its jth-nearest neighbor.
	distances_joint : numpy.array
			The distances to the k-nearest neighbors of each of
			the evaluation points, in the joint space.
	d : int
			The dimension of the marginal space, equivalent to
			the model order.
	Lp_norm : float
			The value of p used to compute the L^{p} norm in 
			computing the distance to nearest neighbors.
			For example, p = 2 (Euclidean norm),
			p = 1 (taxicab), p = np.infty (max-norm),
			etc.

	Returns
	-------
	ter : float
			The estimated total entropy rate using the
			difference of the kth-nearest neighbor 
			estimates.
	ler : numpy.array
			The estimated local entropy rates computed
			as the difference of the local entropies.

	"""
	N = distances_marg.shape[0]

	cd = numpy.power(2*gamma(1 + 1/float(Lp_norm)), (d+1))/gamma(1 + float(d+1)/Lp_norm)

	h_joint = (d+1) * numpy.log(distances_joint[:, n_neighbors-1]) + numpy.log(cd)

	cd = numpy.power(2*gamma(1 + 1/float(Lp_norm)), d)/gamma(1 + float(d)/Lp_norm)

	h_marg = d * numpy.log(distances_marg[:, n_neighbors-1]) + numpy.log(cd)

	ler = h_joint - h_marg

	ter = numpy.mean(ler)

	return ter, ler

def estimate_ter_1d(n_neighbors, distances, Lp_norm):
	"""
	Estimate the total entropy of a time series viewed marginally
	using the standard asymptotically unbiased estimator based
	on k-nearest neighbors:

		\hat{h}[X] = \log(n) - \psi(k) + \log V_{d, q} + \frac{d}{n} \sum_{i = 1}^{n} \log \rho_{i, k, q}

	Parameters
	----------
	n_neighbors : int
			The number of the nearest neighbor to compute the distance
			to, i.e. k for a kth-nearest neighbor estimator.
	distances : numpy.array
			The distances to the k-nearest neighbors of each of
			the evaluation points, in the marginal space.
			Each row corresponds to  an evaluation point, and 
			each column corresponds to a jth-nearest neighbor.
			For example,
				distances_marg[i, j]
			is the distance between the ith evaluation point and
			its jth-nearest neighbor.
	Lp_norm : float
			The value of p used to compute the L^{p} norm in 
			computing the distance to nearest neighbors.
			For example, p = 2 (Euclidean norm),
			p = 1 (taxicab), p = np.infty (max-norm),
			etc.

	Returns
	-------
	ter : float
			The estimated total entropy.
	ler : numpy.array
			The estimated local entropy.

	"""

	d = 1
	n = distances.shape[0]
	cd = numpy.power(2*gamma(1 + 1/float(Lp_norm)), d)/gamma(1 + float(d)/Lp_norm)

	ler = numpy.log(n) - digamma(n_neighbors) + numpy.log(cd) + d * numpy.log(distances[:, n_neighbors-1])

	ter = numpy.mean(ler)

	return ter, ler

def estimate_ler_insample(x, p_opt, pow_neighbors = 0.75, n_neighbors = None, is_multirealization = False):
	"""
	Estimate the local entropy rate using distances between kth-nearest
	neighbors in the marginal and joint spaces, using the estimator

	\hat{h}[X_{t} | X_{t - d}^{t - 1}] = \hat{h}[X_{t - d}^{t}] - \hat{h}[X_{t - d}^{t - 1}],

	e.g. the conditional local entropy equals the joint local entropy minus
	the marginal local entropy, where the joint and marginal entropies are 
	estimated separately via kth-nearest neighbor estimators.

	Parameters
	----------
	x : list
			The time series used to estimate the specific
			entropy rate.

	p_opt : int
			The model order used to estimate ler and ser.

	pow_neighbors : float
			A value in [0, 1] that determines the number of
			nearest neighbors to use in estimating the 
			the local entropy rate.
	n_neighbors : int
			The number of nearest neighbors to use.
	is_multirealization : boolean
			Is the time series a single realization or 
			multiple realizations?

	Returns
	-------
	ler : numpy.array
			The local entropy rate estimated using kth-nearest
			neighbor estimators.

			Will be length len(x) - p_opt.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	Lp_norm = 2

	X = embed_ts(x, p_opt, is_multirealization = is_multirealization)

	if n_neighbors is None:
		n_neighbors = int(numpy.ceil(numpy.power(X.shape[0] - 1, pow_neighbors)))

	if p_opt == 0:
		distances, nn_inds = compute_nearest_neighbors_1d(X, n_neighbors, Lp_norm = Lp_norm)

		er_knn, ler_knn = estimate_ter_1d(n_neighbors, distances, Lp_norm)
	else:
		distances_marg, distances_joint, nn_inds_marg, nn_inds_joint = compute_nearest_neighbors(X, n_neighbors, Lp_norm = Lp_norm)

		er_knn, ler_knn = estimate_ter(n_neighbors, distances_marg, distances_joint, p_opt, Lp_norm)

	return ler_knn

def estimate_lce_insample(X, pow_neighbors = 0.75, n_neighbors = None):
	"""
	Estimate the local conditional entropy using distances between kth-nearest
	neighbors in the marginal and joint spaces, using the estimator

	\hat{h}[Y | C] = \hat{h}[C, Y] - \hat{h}[C],

	e.g. the conditional local entropy equals the joint local entropy minus
	the marginal local entropy, where the joint and marginal entropies are 
	estimated separately via kth-nearest neighbor estimators.

	At present, this assumes that Y is a scalar and C is a vector or scalar.

	Parameters
	----------
	X : numpy.array
			The data matrix where the final / right-most column 
			corresponds to Y and the preceding columns
			correspond to C.
	pow_neighbors : float
			A value in [0, 1] that determines the number of
			nearest neighbors to use in estimating the 
			the local entropy rate.
	n_neighbors : int
			The number of neighbors to use. Defaults to
			using the value calculated from pow_neighbors
			unless specified
	is_multirealization : boolean
			Is the time series a single realization or 
			multiple realizations?

	Returns
	-------
	lce : numpy.array
			The local entropy rate estimated using kth-nearest
			neighbor estimators.

			Will have shape (X.shape[0], ).

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	Lp_norm = 2

	if n_neighbors is None:
		n_neighbors = int(numpy.ceil(numpy.power(X.shape[0] - 1, pow_neighbors)))
	# else:
	# 	print("Warning: Overriding pow_neighbors to take n_neighbors = {}".format(n_neighbors))

	distances_marg, distances_joint, nn_inds_marg, nn_inds_joint = compute_nearest_neighbors(X, n_neighbors, Lp_norm = Lp_norm)

	er_knn, ler_knn = estimate_ter(n_neighbors, distances_marg, distances_joint, X.shape[1] - 1, Lp_norm)

	return ler_knn

def estimate_ser_insample(x, ler, p_opt, q = 0, pow_neighbors = 0.75, is_multirealization = False):
	"""
	Estimate the specific entropy rate in-sample by smoothing
	the local entropy rate estimates against the delay vectors
	using a k-nearest neighbor smoother.

	NOTE: ler should be computed using the same value of 
	p as p_opt, so ler should have length len(x) - p_opt.

	Parameters
	----------
	x : list
			The time series used to estimate the specific
			entropy rate.

	ler : numpy.array
			The local entropy rate estimated using p_opt,
			so should be length len(x) - p_opt.

	p_opt : int
			The model order used to estimate ler and ser.

	q : int
			The predictive horizon that the local entropy rate
			was estimated over and the specific entropy rate
			should be estimated over. Should be >= 0.

	pow_neighbors : float
			A value in [0, 1] that determines the number of
			nearest neighbors to use in the regression of
			the local entropy rate on the delay vectors.

	Returns
	-------
	ser : numpy.array
			The specific entropy rate estimated via 
			regression of the local entropy rate on the
			delay vectors. 

			Will be length len(x) - p_opt.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	if q == 0:
		X   = embed_ts(x, p_opt, is_multirealization)
	else:
		X   = embed_ts_multihorizon(x, p_opt, q, is_multirealization)

	assert len(ler) == X.shape[0], "Error: Use estimated local entropy rate (ler) with the same model order p_opt and predictive horizon q."

	n_neighbors = int(numpy.ceil(numpy.power(X.shape[0] - 1, pow_neighbors)))

	knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')

	knn_out = knn.fit(X[:, :-1], ler)

	ser = knn_out.predict(X[:, :-1])

	return ser

def estimate_normalized_qstep_insample(x, p_opt, q, pow_neighbors = 0.75, return_log_dist_ratio = False):
	"""
	Estimate the specific entropy rate in-sample by smoothing
	the local entropy rate estimates against the delay vectors
	using a k-nearest neighbor smoother.

	NOTE: ler should be computed using the same value of 
	p as p_opt, so ler should have length len(x) - p_opt.

	Parameters
	----------
	x : list
			The time series used to estimate the normalized q-step specific entropy rate.

	p_opt : int
			The model order used to estimate the normalized q-step specific entropy rate.

	q : int
			The predictive horizon for the normalized q-step entropy rate. Should be >= 0.

	pow_neighbors : float
			A value in [0, 1] that determines the number of
			nearest neighbors to use in the regression of
			the local entropy rate on the delay vectors.

	Returns
	-------
	kl_q : numpy.array
			The normalized q-step specific entropy rate estimated as the Kullback-Leibler divergence between the 0-step predictive density and the q-step predictive density with the 0-step predictive density as basline.

			Will be len(x) - p_opt - q.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	X_0 = embed_ts_multihorizon(x, p_opt, 0)
	X_q = embed_ts_multihorizon(x, p_opt, q)

	X_0 = X_0[:-q, :]

	n_neighbors_for_den = 5

	# Note: Below, in compute_nearest_neighbors_cross, we use 
	# one additional nearest neighbor when the first two arguments
	# are the same since this will include a 0-nearest neighbor
	# that is the point itself giving a distance of 0.

	# Using 0-step future as baseline density:
	distances_0 = compute_nearest_neighbors_cross(X_0, X_0, n_neighbors_for_den+1, Lp_norm = 2)
	distances_q = compute_nearest_neighbors_cross(X_q, X_0, n_neighbors_for_den, Lp_norm = 2)

	dist_ratio = distances_q[:, -1]/distances_0[:, -1]

	# Using q-step future as baseline density:
	# Using future as baseline density:
	# distances_0 = compute_nearest_neighbors_cross(X_q, X_q, n_neighbors_for_den+1, Lp_norm = 2)
	# distances_q = compute_nearest_neighbors_cross(X_0, X_q, n_neighbors_for_den, Lp_norm = 2)
	#
	# dist_ratio = distances_0[:, -1]/distances_q[:, -1]

	log_dist_ratio = (p_opt + 1)*numpy.log(dist_ratio)

	n_neighbors_for_reg = int(numpy.ceil(numpy.power(X_0.shape[0] - 1, pow_neighbors)))

	knn = neighbors.KNeighborsRegressor(n_neighbors_for_reg, weights='uniform')

	knn_out = knn.fit(X_0[:, :-1], log_dist_ratio)

	kl_q = knn_out.predict(X_0[:, :-1])

	if return_log_dist_ratio == True:
		return kl_q, log_dist_ratio
	else:
		return kl_q

def estimate_ser_outsample(x_train, x_test, ler_train, p_opt, q = 0, pow_neighbors = 0.75):
	"""
	Estimate the specific entropy rate out-of-sample by
	evaluating the regression function estimated using x_train
	and ler_train at the points given by the delay vectors of
	x_test.

	NOTE: ler_train should be computed using the same value of 
	p as p_opt, so ler_train should have length len(x_train) - p_opt.

	Parameters
	----------
	x_train : list
			The in-sample time series used to estimate
			the specific entropy rate.

	x_test : list
			The out-of-sample time series used to
			evaluate the specific entropy rate estimator.

	ler_train : numpy.array
			The in-sample local entropy rate estimated
			using p_opt with x_train.

	p_opt : int
			The model order used to estimate ler_train and ser.

	q : int
			The predictive horizon that the local entropy rate
			was estimated over and the specific entropy rate
			should be estimated over. Should be >= 0.

	pow_neighbors : float
			A value in [0, 1] that determines the number of
			nearest neighbors to use in the regression of
			the local entropy rate on the delay vectors.

	Returns
	-------
	ser : numpy.array
			The specific entropy rate estimated via 
			regression of the local entropy rate on the
			delay vectors. 

			Will be length len(x) - p_opt.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	assert len(ler_train) == len(x_train) - (p_opt + q), "Error: Use estimated local entropy rate (ler_train) with the same model order p_opt and predictive horizon q."

	if q == 0:
		X_train = embed_ts(x_train, p_opt)
		X_test	= embed_ts(x_test, p_opt)
	else:
		X_train = embed_ts_multihorizon(x_train, p_opt, q)
		X_test	= embed_ts_multihorizon(x_test, p_opt, q)

	n_neighbors = int(numpy.ceil(numpy.power(X_train.shape[0] - 1, pow_neighbors)))

	knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')

	knn_out = knn.fit(X_train[:, :-1], ler_train)

	ser = knn_out.predict(X_test[:, :-1])

	return ser

def estimate_normalized_qstep_outsample(x_train, x_test, log_dist_ratio_train, p_opt, q, pow_neighbors = 0.75):
	"""
	Estimate the normalized q-step specific entropy rate 
	out-of-sample by evaluating the regression function estimated
	using x_train and ler_train at the points given by the delay
	vectors of x_test.

	NOTE: log_dist_ratio_train should be computed using the same value of 
	p as p_opt, so log_dist_ratio_train should have length len(x_train) - p_opt.

	Parameters
	----------
	x_train : list
			The in-sample time series used to estimate
			the specific entropy rate.

	x_test : list
			The out-of-sample time series used to
			evaluate the specific entropy rate estimator.

	log_dist_ratio_train : numpy.array
			The in-sample log of the ratio of distances estimated
			using p_opt with x_train.

	p_opt : int
			The model order used to estimate ler_train and ser.

	q : int
			The predictive horizon that the local entropy rate
			was estimated over and the specific entropy rate
			should be estimated over. Should be >= 0.

	pow_neighbors : float
			A value in [0, 1] that determines the number of
			nearest neighbors to use in the regression of
			the local entropy rate on the delay vectors.

	Returns
	-------
	kl_q : numpy.array
			The normalized q-step specific entropy rate estimated as the Kullback-Leibler divergence between the 0-step predictive density and the q-step predictive density with the 0-step predictive density as basline.

			Will be len(x_test) - p_opt - q.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	X_0 = embed_ts_multihorizon(x_train, p_opt, 0)

	X_0_test = embed_ts_multihorizon(x_test, p_opt, 0)

	X_0      = X_0[:-q, :]

	n_neighbors_for_den = 5

	n_neighbors_for_reg = int(numpy.ceil(numpy.power(X_0.shape[0] - 1, pow_neighbors)))

	knn = neighbors.KNeighborsRegressor(n_neighbors_for_reg, weights='uniform')

	knn_out = knn.fit(X_0[:, :-1], log_dist_ratio_train)

	kl_q = knn_out.predict(X_0_test[:, :-1])

	return kl_q

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

def loocv_mse_joint(n_neighbors, neighbor_inds, X):
	"""
	loocv_mse_joint computes the mean-squared error
	of a k-nearest predictor for a joint process using 
	leave-one-crossvalidation, where the final two
	columns of X correspond to the joint future.

	loocv_mse_joint is set up to run with nlopt.

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
			in the two right-most columns.

	Returns
	-------
	mse : float
			The leave-one-out cross-validated mean-squared
			error of the k-nearest neighbor predictor.
	"""

	n_neighbors = int(numpy.floor(n_neighbors))

	errY = numpy.power(X[:, -2] - numpy.mean(X[neighbor_inds[:, :n_neighbors], -2], 1), 2)
	errX = numpy.power(X[:, -1] - numpy.mean(X[neighbor_inds[:, :n_neighbors], -1], 1), 2)
	mseY = numpy.mean(errY)
	mseX = numpy.mean(errX)

	mse = mseY + mseX
	
	# print s, n_neighbors, mse

	return mse

def smooth(x,window_len=11,window='hanning'):
	"""smooth the data using a window with requested size.
	
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.

	From: http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
	
	input:
		x: the input signal 
		window_len: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			flat window will produce a moving average smoothing.

	output:
		the smoothed signal
		
	example:

	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)
	
	see also: 
	
	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
	scipy.signal.lfilter
 
	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""

	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")

	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")


	if window_len<3:
		return x


	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


	s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=numpy.ones(window_len,'d')
	else:
		w=eval('numpy.'+window+'(window_len)')

	y=numpy.convolve(w/w.sum(),s,mode='valid')
	return y

def estimate_ais(x, p, n_neighbors = 5):
	"""
	Estimate the active information storage between the future and
	a past of length p_opt,

		$A(p) = I[X_{t} ^ X_{t-p}^{t-1}]$

	We use the Java Information Dynamics Toolbox (JIDT) to estimate
	the active information storage, using the KSG-style mutual
	information estimator.

	Parameters
	----------
	x : numpy.array
			The time series
	p : int
			The order of the active information storage to estimate.
	k : int
			The number of nearest neighbors to use in estimating
			the local transfer entropy.

	Returns
	-------
	ais_estimate : float
			The estimated active information storage,
			using the second KSG estimator.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Initialize JIDT:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	jarLocation = '../jidt/infodynamics.jar'

	if not isJVMStarted():
		startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

	implementingClass = "infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov2"
	indexOfLastDot = str.rfind(implementingClass, ".")
	implementingPackage = implementingClass[:indexOfLastDot]
	implementingBaseName = implementingClass[indexOfLastDot+1:]
	miCalcClass = eval('JPackage(\'%s\').%s' % (implementingPackage, implementingBaseName))
	miCalc = miCalcClass()

	# Turn off the (artificial) addition of 
	# observational noise:

	miCalc.setProperty("NOISE_LEVEL_TO_ADD", "0")

	# Set the nearest neighbor parameter:

	miCalc.setProperty("k", "{}".format(n_neighbors))

	X_ais = embed_ts(x, p_max = p)

	sourceArray = X_ais[:, :-1]
	destArray = X_ais[:, -1].reshape(-1, 1)

	miCalc.initialise(sourceArray.shape[1], 1)

	miCalc.setObservations(sourceArray, destArray)

	ais_estimate = miCalc.computeAverageLocalOfObservations()

	return ais_estimate

def estimate_mi(X, Y, n_neighbors = 5):
	"""
	Estimate the mutual information between X and Y, using the 
	Java Information Dynamics Toolbox (JIDT) implementation of
	the KSG-style mutual information estimator.

	Parameters
	----------
	X : numpy.array
			Realizations from the first random variable.
	Y : numpy.array
			Realizations from the second random variable.
	n_neighbors : int
			The number of nearest neighbors to use in estimating
			the local transfer entropy.

	Returns
	-------
	mi_estimate : float
			The estimated mutual information between X and Y,
			using the second KSG estimator.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Initialize JIDT:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	jarLocation = '../jidt/infodynamics.jar'

	if not isJVMStarted():
		startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

	implementingClass = "infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov2"
	indexOfLastDot = str.rfind(implementingClass, ".")
	implementingPackage = implementingClass[:indexOfLastDot]
	implementingBaseName = implementingClass[indexOfLastDot+1:]
	miCalcClass = eval('JPackage(\'%s\').%s' % (implementingPackage, implementingBaseName))
	miCalc = miCalcClass()

	# Turn off the (artificial) addition of 
	# observational noise:

	miCalc.setProperty("NOISE_LEVEL_TO_ADD", "0")

	# Set the nearest neighbor parameter:

	miCalc.setProperty("k", "{}".format(n_neighbors))

	sourceArray = X.reshape(-1, 1)
	destArray = Y.reshape(-1, 1)

	miCalc.initialise(sourceArray.shape[1], destArray.shape[1])

	miCalc.setObservations(sourceArray, destArray)

	mi_estimate = miCalc.computeAverageLocalOfObservations()

	return mi_estimate

def estimate_lte(y, x, q, p, delay, k = 5):
	#tmp_lte, tmp_tte = estimate_lte(y, x, q_IO, p_O, delay = 0, k = 5)
	"""
	Estimate the local transfer entropy from y to x with autoregressive
	order q for y and p for x, and a time delay from y to x of delay.

	The local transfer entropy is the expectand of the
	total transfer entropy, which is the mutual information
	between the future of X and the past of Y, conditional on
	the past of X,

		$I[X_{t}; Y_{t-q-delay}^{t-1-delay} | X_{t-p}^{t-1}]$

	We use the Java Information Dynamics Toolbox (JIDT) to estimate
	the local transfer entropy, using the KSG-inspired conditional 
	mutual information estimator.

	Parameters
	----------
	y : numpy.array
			The nominal input process.
	x : numpy.array
			The nominal output process.
	q : int
			The autoregressive order for the nominal input process.
	p : int
			The autoregressive order for the nominal output process.
	delay : int
			The time delay to use for the input process, where
			delay = 0 would give the standard (non-delayed) 
			transfer entropy.
	k : int
			The number of nearest neighbors to use in estimating
			the local transfer entropy.

	Returns
	-------
	r : int
			description

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	# Automatically convert to multi-trial/realization
	# representation, even if x is just a single trial.

	if len(x.shape) == 1:
		x = x.reshape(1, -1)
		y = y.reshape(1, -1)

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Initialize the Java interface with JIDT using
	# JPype:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# NOTE: Need infodynamics.jar in the appropriate directory for this to work!

	jarLocation = "../jidt/infodynamics.jar"

	if not isJVMStarted():
		startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

	implementingClass = "infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorKraskov"
	indexOfLastDot = str.rfind(implementingClass, ".")
	implementingPackage = implementingClass[:indexOfLastDot]
	implementingBaseName = implementingClass[indexOfLastDot+1:]
	miCalcClass = eval('JPackage(\'%s\').%s' % (implementingPackage, implementingBaseName))
	miCalc = miCalcClass()

	# Set the properties of the JIDT LTE calculator based on the input
	# to this function.

	miCalc.setProperty("NOISE_LEVEL_TO_ADD", "0")

	# miCalc.getProperty(miCalcClass.L_PROP_NAME)
	# miCalc.getProperty(miCalcClass.K_PROP_NAME)

	miCalc.setProperty(miCalcClass.L_PROP_NAME, "{}".format(q))
	miCalc.setProperty(miCalcClass.K_PROP_NAME, "{}".format(p))

	# Add 1 here because JIDT uses the convention that delay = 1 corresponds
	# to the standard transfer entropy, while sidpy uses the convention that
	# delay = 0 corresponds to the standard transfer entropy.

	miCalc.setProperty(miCalcClass.DELAY_PROP_NAME, "{}".format(delay+1))
	miCalc.setProperty("k", "{}".format(k))

	miCalc.initialise()

	miCalc.startAddObservations()

	for trial in range(0,x.shape[0]):
		miCalc.addObservations(y[trial, :], x[trial, :])

	miCalc.finaliseAddObservations()

	lTEs=miCalc.computeLocalOfPreviousObservations()[:]
	TE = numpy.nanmean(lTEs)

	# Need to handle the case of single-realization and
	# multi-realization, since JIDT handles these two
	# cases differently.

	# For a single realization of length T, 
	# JIDT returns a length T array with the first 
	# r = max {p, q + delay} set to 0.

	# For a multi-realization, where each of N
	# realizations has length T, JIDT returns
	# a length N*(T - r) array that concatenates
	# the estimated local transfer entropies.

	r = numpy.max([p, q + delay])

	if x.shape[0] == 1:
		lTEs[:r] = numpy.nan
	else:
		lTEs = stack_sid_by_trial(lTEs, r, num_trials = x.shape[0], points_per_trial = x.shape[1])

	return lTEs, TE

def estimate_lte_iopo(y, x, q, p_io, p_o, delay, k = 5):
	"""
	Estimate the local transfer entropy from y to x with autoregressive
	order q for y and p for x, and a time delay from y to x of delay.

	The local transfer entropy is the expectand of the
	total transfer entropy. Here, we estimate the local entropy rates
	for the output-only and input-output autorgressive model separately,
	rather than use the conditional mutual information definition
	in the SPO formulation of transfer entropy.
	

	Parameters
	----------
	y : numpy.array
			The nominal input process.
	x : numpy.array
			The nominal output process.
	q : int
			The autoregressive order for the nominal input process.
	p_io : int
			The autoregressive order for the nominal output process
			when including the input.
	p_o : int
			The autoregressive order for the nominal output process
			when excluding the input.
	delay : int
			The time delay to use for the input process, where
			delay = 0 would give the standard (non-delayed) 
			transfer entropy.
	k : int
			The number of nearest neighbors to use in estimating
			the local transfer entropy.

	Returns
	-------
	r : int
			description

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	# Automatically convert to multi-trial/realization
	# representation, even if x is just a single trial.

	if len(x.shape) == 1:
		x = x.reshape(1, -1)
		y = y.reshape(1, -1)

	r = numpy.max([q + delay, p_io, p_o])

	Y = embed_ts(y, r, is_multirealization = True)
	X = embed_ts(x, r, is_multirealization = True)

	Yt = Y[:, r - q:r]
	Xt = X[:, r - p_io:]

	Z = numpy.concatenate((Yt, Xt), 1)

	ler_io = estimate_lce_insample(Z, pow_neighbors = None, n_neighbors = k)

	Z = X[:, r - p_o:]

	ler_o =  estimate_lce_insample(Z, pow_neighbors = None, n_neighbors = k)

	lTEs = ler_o - ler_io

	lTEs = stack_sid_by_trial(lTEs, r, num_trials = x.shape[0], points_per_trial = x.shape[1])

	TE = numpy.nanmean(lTEs)

	return lTEs, TE

def determine_delay(y, x, p, q = 1, method = 'maxTE', verbose = False):
	"""
	Description of function goes here

	Parameters
	----------
	y : numpy.array
			The nominal input process.
	x : numpy.array
			The nominal output process.
	p : int
			The autoregressive order for the nominal output process.
	q : int
			The autoregressive order for the nominal input process.
			Defaults to 1 for determining the optimal delay using JIDT.
	method : string
			What method to use in determining the optimal delay. One of
			{'maxTE', 'minMSE'}.
	verbose : boolean
			Whether to announce the stages of determine_delay.

	Returns
	-------
	delay : int
			The optimized delay for the input process.
			delay = 0 corresponds to standard Transfer entropy.
	lTEs : numpy.array
			The local transfer entropies corresponding to the
			optimized delay.
	delays : numpy.array
			The delays considered by determine_delay.
	TE_by_delay : float
			The total transfer entropies estimated using
			the delays considered by determine_delay

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	# Automatically convert to multi-trial/realization
	# representation, even if x is just a single trial.

	if len(x[0, :].shape):
		x = x.reshape(1, -1)
		y = y.reshape(1, -1)


	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Initialize the Java interface with JIDT using
	# JPype:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# NOTE: Need infodynamics.jar in the appropriate directory for this to work!

	jarLocation = "../jidt/infodynamics.jar"

	if not isJVMStarted():
		startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

	implementingClass = "infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorKraskov"
	indexOfLastDot = str.rfind(implementingClass, ".")
	implementingPackage = implementingClass[:indexOfLastDot]
	implementingBaseName = implementingClass[indexOfLastDot+1:]
	miCalcClass = eval('JPackage(\'%s\').%s' % (implementingPackage, implementingBaseName))
	miCalc = miCalcClass()

	miCalc.setProperty("NOISE_LEVEL_TO_ADD", "0")

	miCalc.setProperty(miCalcClass.L_PROP_NAME, "{}".format(q))
	miCalc.setProperty(miCalcClass.K_PROP_NAME, "{}".format(p))

	delays = list(range(1, 11))
	TE_by_delay = numpy.zeros(len(delays))

	lTEs_by_delay = {}

	if verbose:
		print('Searching for optimal delay using delay* = argmin TE(delay)...')
	for delay_ind, delay in enumerate(delays):
		if verbose:
			print('On delay = {}...'.format(delay))

		miCalc.setProperty(miCalcClass.DELAY_PROP_NAME, "{}".format(delay))

		miCalc.initialise()

		# print 'Adding observations...'

		miCalc.startAddObservations()

		for trial in range(0,x.shape[0]):
			miCalc.addObservations(y[trial, :], x[trial, :])

		miCalc.finaliseAddObservations()

		# print 'Computing local Transfer Entropies...'

		lTEs=miCalc.computeLocalOfPreviousObservations()[:]
		TE = numpy.mean(lTEs)

		TE_by_delay[delay_ind] = TE

		r = numpy.max([p, q + delay - 1])

		if x.shape[0] == 1:
			lTEs[:r] = numpy.nan
		else:
			lTEs = stack_sid_by_trial(lTEs, q, p, delay - 1, num_trials = x.shape[0], points_per_trial = x.shape[1])

		lTEs_by_delay[delay_ind] = lTEs

	ind = numpy.argmax(TE_by_delay)
	delay = delays[ind] - 1

	lTEs = lTEs_by_delay[ind]

	if verbose:
		print('Chose delay* = {}...'.format(delay))

	return delay, lTEs, delays, TE_by_delay

def stack_io(y, x, q, p, delay):
	"""
	stack_io stacks the input y and output x in a 
	standard data matrix for use in regressions for
	model selection and estimation of specific
	transfer entropy.

	Parameters
	----------
	y : numpy.array
			The nominal input process.
	x : numpy.array
			The nominal output process.
	q : int
			The autoregressive order for the nominal input process.
	p : int
			The autoregressive order for the nominal output process.
	delay : int
			The time delay to use for the input process, where
			delay = 0 would give the standard (non-delayed) 
			transfer entropy, delay = -1 gives a
			contemporaneous transfer entropy, and delay > 0
			gives a delayed transfer entropy.
			

	Returns
	-------
	Z : numpy.array
			The nominal input / output time series
			stacked as to have the input time series
			as the first q columns and the output
			time series as the last p + 1 columns,
			with appropriate time shifts specified
			by delay.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	if len(x.shape) == 1:
		x = x.reshape(1, -1)
		y = y.reshape(1, -1)

	assert delay >= -1, "Error: The delay must be >= -1."

	r = numpy.max([p, q + delay])

	Y = embed_ts(y, p_max = r, is_multirealization = True)
	X = embed_ts(x, p_max = r, is_multirealization = True)

	Y_stack = Y[:, r-(q+delay):r-delay]
	X_stack = X[:, r-p:]

	Z = numpy.concatenate((Y_stack, X_stack), 1)

	return Z

def stack_sid_by_trial(sid, r, num_trials, points_per_trial, is_regular = True):
	"""
	stack_sid_by_trial stacks the specific information dynamic measure
	in a trial-by-trial / realization-by-realization numpy array. This
	is necessary because regression-based estimates of STE will return
	all of the trials collapsed into a single long pseudo-trial.

	Parameters
	----------
	sid : numpy.array
			The specific information dynamic measure to reformat
			into the proper trial-by-trial numpy.array.
	r : int
			The maximum lag to use, typically 
			max {p_io, p_o, q + delay},
	num_trials : int
			The number of trials / realizations used to estimate the
			information dynamical quantity.
	points_per_trial : int or numpy.array
			The number of points per trial for which the quantity is
			estimated.

			If is_regular = True, points_per_trial is an int
			giving the value for all trials.

			If is_regular = False, points_per_trial is a numpy.array
			giving the value for each trial.
	is_regular : boolean
			Whether each trial is the same length (True) or not (False).	

	Returns
	-------
	sid_by_trial : numpy.array
			The specific information dynamic measure formatted
			into a num_trials X points_per_trial array appropriate
			for direct comparison to the original time series.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	if is_regular:
		points_per_truncated_trial = points_per_trial - r

		sid_by_trial = numpy.empty((num_trials, points_per_truncated_trial + r))

		sid_by_trial.fill(numpy.nan)

		for trial_ind in range(num_trials):
			sid_by_trial[trial_ind, r:] = sid[trial_ind*points_per_truncated_trial:(trial_ind + 1)*points_per_truncated_trial]
	else:
		points_per_truncated_trial = numpy.array(points_per_trial) - r

		sid_by_trial = []

		for trial_ind in range(num_trials):
			cur_trial = numpy.empty(points_per_trial[trial_ind])
			cur_trial.fill(numpy.nan)

			cur_trial[r:] = sid[numpy.sum(points_per_truncated_trial[:trial_ind]):numpy.sum(points_per_truncated_trial[:trial_ind+1])]

			sid_by_trial.append(cur_trial)

	return sid_by_trial

def estimate_ste(y, x, q, p, delay, lTEs, pow_neighbors = 0.5, verbose = False):
	"""
	estimate_ste estimates the specific transfer entropy by smoothing
	the local transfer entropy estimates against the delay vectors
	using a k-nearest neighbor smoother.

	Parameters
	----------
	y : numpy.array
			The nominal input process.
	x : numpy.array
			The nominal output process.
	q : int
			The autoregressive order for the nominal input process.
	p : int
			The autoregressive order for the nominal output process.
	delay : int
			The time delay to use for the input process, where
			delay = 0 would give the standard (non-delayed) 
			transfer entropy, delay = -1 gives a
			contemporaneous transfer entropy, and delay > 0
			gives a delayed transfer entropy.
	lTEs : numpy.array
			The estimated lTEs returned by estimate_lte.
	pow_neighbors : float
			A value in [0, 1] that determines the number of
			nearest neighbors to use in the regression of
			the local transfer entropy on the delay vectors.
	verbose : boolean
			Whether or not to announce the steps in estimate_ste.
			

	Returns
	-------
	sTEs_by_trial : numpy.array
			The estimated specific transfer entropies per-trial.
	lTEs_by_trial : numpy.array
			The estimated local transfer entropies per-trial.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	if verbose:
		print('Storing LTEs by trial...')

	r = numpy.max([p, q + delay])

	if len(x.shape) == 1:
		x = x.reshape(1, -1)
		y = y.reshape(1, -1)

	if len(lTEs.shape) == 1:
		lTEs  = embed_ts(lTEs, r, is_multirealization = False)
	else:
		lTEs  = embed_ts(lTEs, r, is_multirealization = True)

	if verbose:
		print('Computing sTEs...')

	Z = stack_io(y, x, q, p, delay)

	n_neighbors = int(numpy.floor(numpy.power(Z.shape[0], pow_neighbors)))

	knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')

	knn_out = knn.fit(Z[:, :-1], lTEs[:, -1])

	sTEs = knn_out.predict(Z[:, :-1])

	points_per_truncated_trial = x.shape[1] - r

	if verbose:
		print('Storing sTEs by trial...')

	r = numpy.max([q + delay, p])

	sTEs_by_trial = stack_sid_by_trial(sTEs, r, num_trials = x.shape[0], points_per_trial = x.shape[1])

	if x.shape[0] == 1:
		sTEs = sTEs_by_trial.flatten()
	else:
		sTEs = sTEs_by_trial

	return sTEs


def estimate_ste_iopo(y, x, q, p_io, p_o, delay, lTEs, pow_neighbors = 0.5, verbose = False):
	"""
	estimate_ste_iopo estimates the specific transfer entropy by 
	smoothing the local transfer entropy estimates against the delay vectors
	using a k-nearest neighbor smoother.

	Because p_io =/= p_o, this can be used with the local transfer entropies
	estimated under the Input-Output-Predictively Optimal (IOPO) formulation
	in addition to under the Self-Predictively Optimal (SPO) formulation.

	Parameters
	----------
	y : numpy.array
			The nominal input process.
	x : numpy.array
			The nominal output process.
	q : int
			The autoregressive order for the nominal input process.
	p_io : int
			The autoregressive order for the nominal output process,
			including the nominal input process.
	p_o : int
			The autoregressive order for the nominal output process,
			excluding the nominal input process.
	delay : int
			The time delay to use for the input process, where
			delay = 0 would give the standard (non-delayed) 
			transfer entropy, delay = -1 gives a
			contemporaneous transfer entropy, and delay > 0
			gives a delayed transfer entropy.
	lTEs : numpy.array
			The estimated lTEs returned by estimate_lte.
	pow_neighbors : float
			A value in [0, 1] that determines the number of
			nearest neighbors to use in the regression of
			the local transfer entropy on the delay vectors.
	verbose : boolean
			Whether or not to announce the steps in estimate_ste.
			

	Returns
	-------
	sTEs_by_trial : numpy.array
			The estimated specific transfer entropies per-trial.
	lTEs_by_trial : numpy.array
			The estimated local transfer entropies per-trial.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	if verbose:
		print('Storing LTEs by trial...')

	r = numpy.max([p_io, p_o, q + delay])

	if len(x.shape) == 1:
		x = x.reshape(1, -1)
		y = y.reshape(1, -1)

	if len(lTEs.shape) == 1:
		lTEs  = embed_ts(lTEs, r, is_multirealization = False)
	else:
		lTEs  = embed_ts(lTEs, r, is_multirealization = True)

	if verbose:
		print('Computing sTEs...')

	Z = stack_io(y, x, q, numpy.max([p_io, p_o]), delay)

	n_neighbors = int(numpy.floor(numpy.power(Z.shape[0], pow_neighbors)))

	knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')

	knn_out = knn.fit(Z[:, :-1], lTEs[:, -1])

	sTEs = knn_out.predict(Z[:, :-1])

	points_per_truncated_trial = x.shape[1] - r

	if verbose:
		print('Storing sTEs by trial...')

	sTEs_by_trial = stack_sid_by_trial(sTEs, r, num_trials = x.shape[0], points_per_trial = x.shape[1])

	if x.shape[0] == 1:
		sTEs = sTEs_by_trial.flatten()
	else:
		sTEs = sTEs_by_trial

	return sTEs

def generate_refined_aaft_surrogate(x, num_iterations = None, seed = None, print_spectra_error = False, to_plot = False):
	"""
	Generate refined amplitude adjusted Fourier transform (AAFT) surrogates,
	following the method from 

	T. Schreiber and A. Schmitz, "Improved Surrogate Data for Nonlinearity Tests,"
	Phys. Rev. Lett., vol. 77, no. 4, pp. 635-638, Jul. 1996.

	The surrogate time series xstar has the same marginal distribution ('amplitudes')
	and approximately the same sample spectral density (periodogram) as the original
	time series x.

	These surrogates are appropriate for testing the null hypothesis that x originates
	from an invertible, instantaneous transformation of a Gaussian colored noise process.

	Parameters
	----------
	x : list or numpy.array
			The original time series used to generate surrogates.
	num_iterations : int
			The number of loops through the AAFT refinement process.
			If None, run until the change in the error is below a
			fixed threshold.
	seed : int
			A seed for numpy's pseudo-random number generator.
			Must be convertible to 32 bit unsigned integers.
			If None, then the current state of the PRNG is
			used.
	print_spectra_error : boolean
			Whether to print the mean-squared error between the
			periodogram of x and the periodogram of xstar.
	to_plot : boolean
			Whether or not to plot the time series and its
			surrogate along with the periodogram for the 
			time series and its surrogate.


	Returns
	-------
	xstar : numpy.array
			The surrogate generated using the refined AAFT
			method.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	spec_change_tol = 0.0000001

	if num_iterations == None:
		num_iterations = 100

	if seed is not None:
		numpy.random.seed(seed)

	spec_orig = numpy.abs(numpy.fft.fft(x))

	# Sort the original time series and compute
	# its Fast Fourier Transform, along with the
	# amplitude and phase of the Fourier transform.

	xsort = numpy.sort(x)

	Xfft = numpy.fft.fft(x)

	Sx = numpy.abs(Xfft)
	Phasex = numpy.angle(Xfft)

	# Demonstrate that we recover the original
	# time series by taking the inverse Fourier
	# transform.

	# Xfft_recover = Sx*numpy.exp(1j*Phasex)
	# 
	# xback = numpy.fft.ifft(Xfft)

	# Generate the first iterate of the 
	# surrogate time series, which is just a
	# complete randomization (sampling with 
	# replacement) of the original time series.

	xstar = numpy.copy(x)

	numpy.random.shuffle(xstar)

	spec_surr = numpy.abs(numpy.fft.fft(xstar))

	spec_error_old = numpy.mean(numpy.power(spec_orig - spec_surr, 2))

	# Iterate the refined AAFT procedure:

	continue_loop = True

	iter_counter = 0

	while continue_loop:
		# Match the Fourier amplitudes of the original
		# and surrogate time series while maintaining
		# the phases.

		Xstarfft = numpy.fft.fft(xstar)

		Phasexstar = numpy.angle(Xstarfft)

		Xstarfft = Sx*numpy.exp(1j*Phasexstar)

		# Recover the surrogate time series from its
		# Fourier transform.

		xstar = numpy.real(numpy.fft.ifft(Xstarfft))

		# if to_plot and (num_iterations - n) <= 1:
		# 	plt.figure()
		# 	plt.plot(x)
		# 	plt.plot(xstar)

		# 	fig, ax = plt.subplots(2)
		# 	ax[0].hist(x, bins = 50)
		# 	ax[1].hist(xstar, bins = 50)

		# 	plt.figure()
		# 	plt.plot(numpy.abs(numpy.fft.fft(x))[1:x.shape[0]//2])
		# 	plt.plot(numpy.abs(numpy.fft.fft(xstar))[1:x.shape[0]//2])

		# Recover the time series amplitudes by using
		# a rank ordering of the surrogate time series.

		order = xstar.argsort()
		ranks = order.argsort()

		xstar = xsort[ranks]

		spec_surr = numpy.abs(numpy.fft.fft(xstar))

		# if to_plot and (num_iterations - n) <= 1:
		# 	plt.figure()
		# 	plt.plot(x)
		# 	plt.plot(xstar)

		# 	fig, ax = plt.subplots(2)
		# 	ax[0].hist(x, bins = 50)
		# 	ax[1].hist(xstar, bins = 50)

		# 	plt.figure()
		# 	plt.plot(numpy.abs(numpy.fft.fft(x))[1:x.shape[0]//2])
		# 	plt.plot(numpy.abs(numpy.fft.fft(xstar))[1:x.shape[0]//2])

		spec_error_new = numpy.mean(numpy.power(spec_orig - spec_surr, 2))

		if print_spectra_error:
			print(spec_error_new)

		if numpy.abs(spec_error_new - spec_error_old)/numpy.abs(spec_error_old) < spec_change_tol or iter_counter == num_iterations:
			continue_loop = False

			if to_plot:
				plt.figure()
				plt.plot(x)
				plt.plot(xstar)

				fig, ax = plt.subplots(2)
				ax[0].hist(x, bins = 'auto')
				ax[1].hist(xstar, bins = 'auto')

				plt.figure()
				plt.plot(spec_orig[1:x.shape[0]//2])
				plt.plot(spec_surr[1:x.shape[0]//2])
		else:
			continue_loop = True
			iter_counter += 1

			spec_error_old = spec_error_new

	return xstar

def embed_ts_multilag(x, dt, Tp, tf, is_multirealization = False):
	"""
	Embed a time series into a data matrix X appropriate
	for an analysis of a sampled continuous-time system
	where one desires to determine the optimal sampling
	for prediction at a given future horizon.

	The goal is to predict the future at a time t + tf,
	given the past values in [t - Tp, t], where the 
	signal is sampled using a sampling rate of dt.

	embed_ts_multilag generates a data matrix that
	is appropriate to use with extract_multilag_from_embed,
	so that data lengths, past-future pairs, etc., are 
	comparable using different values of a lower sampling
	rate Dt = m*dt.

	Parameters
	----------
	x : numpy.array
			The time series to embed in a multi-lag
			data matrix.
	dt : float
			The sampling rate for the signal.
	Tp : float
			The total length of the interval into the
			past to consider when predicting the future.
	tf : float
			The amount of time into the future, relative
			to the present, to predict.

	Returns
	-------
	X : numpy.array
			The data matrix, for use with
			extract_multilag_from_embed.

	"""

	Mp = int(Tp/float(dt))

	n = int(tf/float(dt))

	p_max = Mp + n

	X = embed_ts(x, p_max, is_multirealization = is_multirealization)

	return X


def extract_multilag_from_embed(X, dt, Tp, tf, dm):
	"""
	Extract the appropriate elements from a data matrix
	created by using embed_ts_multilag with these same
	parameters.


	Parameters
	----------
	X : numpy.array
			The data matrix created using embed_ts_multilag
			with the same dt, Tp, and tf.
	dt : float
			The sampling rate for the signal.
	Tp : float
			The total length of the interval into the
			past to consider when predicting the future.
	tf : float
			The amount of time into the future, relative
			to the present, to predict.
	dm : int
			The lower sampling rate given by Dt = m*dt.

	Returns
	-------
	Xpf : numpy.array
			The numpy downsampled X matrix, keeping the
			present at t, the future at t + tf, and 
			extracting the relevant elements of the past
			at the new sampling rate Dt = m*dt.
			
	"""

	Mp = int(Tp/float(dt))

	n = int(tf/float(dt))

	dtp = dm*dt

	Xf = X[:, -1]

	Xpresent = X[:, -(n+1)]

	m = int(numpy.floor(Tp/dtp))

	inds = numpy.arange(Mp, Mp-(m+1)*dm,-dm)[::-1]
	Xpast = X[:, inds]

	# print("\n")

	# print(numpy.column_stack((Xpresent, Xf)))

	# print(numpy.column_stack((Xpast, Xf)))

	Xpf = numpy.column_stack((Xpast, Xf))

	return Xpf