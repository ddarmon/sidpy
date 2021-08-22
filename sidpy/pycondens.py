import numpy
import scipy
from scipy.integrate import quad

from sklearn.metrics.pairwise import pairwise_distances

from scipy.stats import norm

import sidpy

from itertools import islice

import os

def stack_distance_matrix(x, p_max, mean_x = 0., sd_x = 1.0, is_multirealization = False, output_verbose = False):
	if is_multirealization:
		ns = []

		for r in range(len(x)):
			ns.append(x[r].shape[0])

		block_limits = [0] + numpy.cumsum(ns).tolist()

		x_flat = numpy.concatenate(x)

		x_flat = (x_flat - mean_x)/(sd_x)

		D = pairwise_distances(x_flat[:,numpy.newaxis], metric = 'l1')

		# Compute the squared distances and multiply by the
		# -1/2 prefactor. NOTE: For now, we assume a Gaussian
		# kernel for the predictive density.

		# Thus, up to the bandwidth (squared) prefactor,
		# these are the terms we sum and exponentiate
		# to get the kernel of interest.

		D = -0.5*D*D

		# Stack the submatrices of D so that we can 
		# easily compute the distances in the *embedding* space.
		# Do this all at once up to the maximum model order that
		# will be considered.

		total_size = numpy.sum(ns)-len(ns)*p_max

		# De_max = numpy.empty(shape = (total_size, total_size, p_max + 1), dtype = 'float32', order = 'C')
		De_max = numpy.zeros(shape = (total_size, total_size, p_max + 1), dtype = 'float32', order = 'C')

		for block_j in range(len(ns)):
			submatrix_index_j = numpy.arange(block_limits[block_j], block_limits[block_j+1]-p_max)
			sub_for_De_j = submatrix_index_j - block_j*p_max
			for block_i in range(len(ns)):
				submatrix_index_i = numpy.arange(block_limits[block_i], block_limits[block_i+1]-p_max)
				sub_for_De_i = submatrix_index_i - block_i*p_max
				for offset_ind in range(0, p_max+1):
					De_max[numpy.ix_(sub_for_De_i, sub_for_De_j, [offset_ind])] = D[numpy.ix_(submatrix_index_i + offset_ind, submatrix_index_j + offset_ind)][:, :, numpy.newaxis]


	else:
		n = len(x)

		x = (x-mean_x)/sd_x

		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#
		# Compute reasonable cutoffs for turning off a lag
		# based on the kernels all being approximately 
		# constant for that bandwidth.
		# 
		# See notes from 151216-b.
		# 
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		eps = 0.1
		rsx = numpy.power(numpy.max(x) - numpy.min(x), 2)

		hx_cutoff = scipy.optimize.brentq(func_h_cutoff, a = 0.1, b = 100, args = (rsx, eps))

		if output_verbose:
			print('Using bandwidth cutoffs hx = {}'.format(hx_cutoff))

		# Compute the L1 distances between each
		# timepoint in x, 
		# 
		# Hence, D is a symmetric n x n matrix.

		D = pairwise_distances(x[:,numpy.newaxis], metric = 'l1')

		# Compute the squared distances and multiply by the
		# -1/2 prefactor. NOTE: For now, we assume a Gaussian
		# kernel for the predictive density.

		# Thus, up to the bandwidth (squared) prefactor,
		# these are the terms we sum and exponentiate
		# to get the kernel of interest.

		D = -0.5*D*D

		# Let p be the autoregressive order, so we embed into
		# (p+1)-space.

		# p_max is the largest p we will consider.

		ps = range(1, p_max + 1)

		# Stack the submatrices of D so that we can 
		# easily compute the distances in the *embedding* space.
		# Do this all at once up to the maximum model order that
		# will be considered.

		De_max = numpy.empty(shape = (n-p_max, n-p_max, p_max + 1), dtype = 'float32', order = 'C')

		submatrix_index = numpy.arange(0, n-p_max)

		for offset_ind in range(0, p_max+1):
			De_max[:, :, offset_ind] = D[submatrix_index + offset_ind, :][:, submatrix_index + offset_ind]

		# Note that De_max is (n-p_max)x(n-p_max)x(p_max+1), with the *future* values stored in
		# 	De_max[:, :, p_max]
		# and the most distant past stored in 
		# 	De_max[:, :, 0]

	return De_max


def choose_model_order_nlpl_kde(x, p_max, save_name, is_multirealization = False, output_verbose = False):
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Set various parameters.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	lwo_halfwidth = 10

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Get constants for rescaling the data to have 
	# sample mean 0 and sample standard deviation 1.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if is_multirealization:
		x_stacked = numpy.concatenate(x)

		mean_x = x_stacked.mean()
		sd_x   = x_stacked.std()
	else:
		mean_x = x.mean()
		sd_x   = x.std()

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute reasonable cutoffs for turning off a lag
	# based on the kernels all being approximately 
	# constant for that bandwidth.
	# 
	# See notes from 151216-b.
	# 
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if is_multirealization:
		eps = 0.1
		rsx = numpy.power(numpy.max(x_stacked) - numpy.min(x_stacked), 2)
	else:
		eps = 0.1
		rsx = numpy.power(numpy.max(x) - numpy.min(x), 2)

	hx_cutoff = scipy.optimize.brentq(func_h_cutoff, a = 0.1, b = 100, args = (rsx, eps))

	if output_verbose:
		print('Using bandwidth cutoffs hx = {}'.format(hx_cutoff))

	De_max = stack_distance_matrix(x, p_max, mean_x = mean_x, sd_x = sd_x, is_multirealization = is_multirealization, output_verbose = output_verbose)

	# Note that De_max is (n-p_max)x(n-p_max)x(p_max+1), with the *future* values stored in
	# 	De_max[:, :, p_max]
	# and the most distant past stored in 
	# 	De_max[:, :, 0]

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute the marginal (i.e. lag-0) entropy.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	De = De_max[:, :, p_max][:, :, numpy.newaxis]

	h = numpy.array(numpy.power(De_max.shape[0], -1./(5)))

	# opt_method = 'powell'
	# opt_method = 'COBYLA'
	# opt_method = 'SLSQP'
	opt_method = 'nelder-mead'

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Options for Powell-like tolerances:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# options_use = {'ftol' : 0.001, 'xtol' : 0.001, 'maxiter' : 1000, 'maxfev' : 1000, 'disp' : True}

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Options for Nelder-Mead-like tolerances:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if output_verbose:
		to_display = True
	else:
		to_display = False

	options_use = {'ftol' : 0.00001, 'xtol' : 0.001, 'maxiter' : 1000, 'maxfev' : 1000, 'disp' : to_display}

	optim_out = scipy.optimize.minimize(score_data_lwo, numpy.sqrt(h), args = (De, lwo_halfwidth), method=opt_method, options = options_use)
	h_opt = optim_out['x']*optim_out['x']

	nlls = [optim_out['fun']]

	# Store the bandwidths by model order in a dictionary for
	# later inspection.

	h = [h_opt]

	if not os.path.exists('bw-saved'):
		os.makedirs('bw-saved')

	save_bandwidth_o(p_max, h[0], numpy.array([0]), sd_x, 'bw-saved/' + save_name + '-' + str(0))

	if output_verbose:
		print(h)

	hs  = {}

	hs[0] = h

	active_set = numpy.array([p_max])

	active_sets = {}

	active_sets[0] = active_set

	ps = range(1, p_max + 1)

	for p in ps:
		if output_verbose:
			print("On p = {}...".format(p))
		active_set = numpy.array([p_max - p] + active_set.tolist())
		
		De = De_max[:, :, active_set] # Pull out the active set from De_max
		
		if type(h) != list:
			h = h.tolist()
		
		# initially set the bandwidth to the asymptotically optimal
		# value, which for p covariates is n^(-1/(p+5))
		
		h = [numpy.power(De_max.shape[0], -1./(p + 5))] + h

		optim_out = scipy.optimize.minimize(score_data_lwo, numpy.sqrt(h), args = (De, lwo_halfwidth), method=opt_method, options = options_use)
		h_opt = optim_out['x']*optim_out['x']
		
		if output_verbose:
			print("For p = {}, h_opt before screening for large bandwidths is:".format(p))
			print(h_opt)
		
		active_set = active_set[numpy.where(h_opt < hx_cutoff)]
		h = h_opt[numpy.where(h_opt < hx_cutoff)]
		
		hs[p] = h
		active_sets[p] = active_set
		
		save_bandwidth_o(p_max, hs[p], active_sets[p], sd_x, 'bw-saved/' + save_name + '-' + str(p))

		nlls += [optim_out['fun']]

		if output_verbose:
			print("For p = {}, h_opt (after screening), active_set, and NLPL(p, h_opt) are:".format(p))
			print(h, active_set, optim_out['fun'])

	h_raw = {}
	active_set_return = {}

	for p in [0] + ps:
		h_cur, active_set_cur = load_bandwidth_o('bw-saved/' + save_name + '-' + str(p), p_max)

		h_raw[p] = h_cur
		active_set_return[p] = active_set_return

	p_opt = numpy.argmin(nlls)

	return p_opt, nlls, h_raw, active_set_return

def score_model_orders_with_saved_bws(x, p_max, save_name, is_multirealization = False):
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Set various parameters.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	lwo_halfwidth = 10

	De_max = stack_distance_matrix(x, p_max, mean_x = 0.0, sd_x = 1.0, is_multirealization = is_multirealization)

	# Note that De_max is (n-p_max)x(n-p_max)x(p_max+1), with the *future* values stored in
	# 	De_max[:, :, p_max]
	# and the most distant past stored in 
	# 	De_max[:, :, 0]

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute the marginal (i.e. lag-0) negative log-
	# predictive likelihood.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	h, active_set = load_bandwidth_o('bw-saved/' + save_name + '-' + str(0), p_max)

	De = De_max[:, :, p_max][:, :, numpy.newaxis]

	nlls = [score_data_lwo(numpy.sqrt(h), De, lwo_halfwidth)]

	hs  = {}

	hs[0] = h

	active_sets = {}

	active_sets[0] = active_set

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute the higher order (lag-p) negative log-
	# predictive likelihoods.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	ps = range(1, p_max + 1)

	for p in ps:
		h, active_set = load_bandwidth_o('bw-saved/' + save_name + '-' + str(p), p)
		
		De = De_max[:, :, p_max - p:][:, :, active_set] # Pull out the active set from De_max

		nlls.append(score_data_lwo(numpy.sqrt(h), De, lwo_halfwidth))

		hs[p] = h
		active_sets[p] = active_set

	p_opt = numpy.argmin(nlls)

	return p_opt, nlls, hs, active_sets

def estimate_ser_kde(x, p_opt, h, active_set, is_multirealization = False):
	lwo_halfwidth = 10

	De_max = stack_distance_matrix(x, p_opt, mean_x = 0.0, sd_x = 1.0, is_multirealization = is_multirealization)

	De = numpy.empty(shape = (De_max.shape[0], De_max.shape[1], len(active_set)), dtype = 'float32', order = 'C')

	for lag_ind, lag_val in enumerate(active_set):
		De[:, :, lag_ind] = De_max[:, :, lag_val]

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Compute the specific entropy rate via integration
	# of the estimator for the predictive density.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	h_squared = h*h

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled = De_scaled / h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:(len(h)-1)].sum(2)

	X = sidpy.embed_ts(x, p_max = p_opt, is_multirealization = is_multirealization)

	X_futures = X[:, -1]

	summands_bottom = numpy.exp(S_bottom)

	spenra = numpy.zeros(De.shape[0])

	if is_multirealization:
		x_stacked = numpy.concatenate(x)
		sd_x = x_stacked.std()
		x_sorted = numpy.sort(x_stacked)
	else:
		sd_x = x.std()
		x_sorted = numpy.sort(X_futures)

	N = De_scaled.shape[0]

	for t in range(N):
		if t % 100 == 0:
			print('On t = {} of {}.'.format(t, N))

		lb = numpy.max([0, t - lwo_halfwidth])
		ub = numpy.min([N-1, t + lwo_halfwidth])

		mask = numpy.array([1]*lb + [0]*(ub-lb+1) + [1]*(N - ub - 1),dtype=numpy.bool)

		S_bottom_cur = S_bottom[t,:]
		summands_bottom_cur = summands_bottom[t, :]

		quad_flogf_out = quad(integrand_flogf_er, x_sorted[0]-5*sd_x, x_sorted[-1]+5*sd_x, epsabs=0.01, args = (X_futures, S_bottom_cur, summands_bottom_cur, h, h_squared, mask))

		spenra[t] = quad_flogf_out[0]

	return spenra

def score_data_lwo(h_sr, De, lwo_halfwidth = 0, print_iterates = False):
	h = h_sr*h_sr
	h_squared = h*h
	
	p_internal = (De.shape[2] - 1)

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled[:] = De_scaled / h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:p_internal].sum(2)
	S_top = S_bottom + De_scaled[:,:,p_internal]

	summands_bottom = numpy.exp(S_bottom)
	summands_top	= numpy.exp(S_top)

	N = De_scaled.shape[0]

	fs = numpy.empty(N, dtype = 'float32')

	for t in range(N):
		lb = numpy.max([0, t - lwo_halfwidth])
		ub = numpy.min([N-1, t + lwo_halfwidth])

		mask = numpy.array([1]*lb + [0]*(ub-lb+1) + [1]*(N - ub - 1),dtype=numpy.bool)

		s_top = summands_top[t, :][mask]
		s_bottom = summands_bottom[t, :][mask]

		fs[t] = s_top.sum()/s_bottom.sum()
	
	fs = fs/h[p_internal]/numpy.sqrt(2*numpy.pi)

	score = -numpy.log(fs).mean()
	
	if print_iterates:
		print(h, score)
	
	return score

def score_data_lwo_bm(h_sr, De, lwo_halfwidth = 0, print_iterates = False):
	h = h_sr*h_sr
	h_squared = h*h
	
	p_internal = (De.shape[2] - 1)

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled /= h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:p_internal].sum(2)
	S_top = S_bottom + De_scaled[:,:,p_internal]

	summands_bottom = numpy.exp(S_bottom)
	summands_top	= numpy.exp(S_top)

	N = De_scaled.shape[0]

	fs = numpy.empty(N, dtype = 'float32')

	for t in range(N):
		lb = numpy.max([0, t - lwo_halfwidth])
		ub = numpy.min([N-1, t + lwo_halfwidth])

		mask = numpy.array([1]*lb + [0]*(ub-lb+1) + [1]*(N - ub - 1),dtype=numpy.bool)

		s_top = numpy.compress(mask, summands_top[t, :])
		s_bottom = numpy.compress(mask, summands_bottom[t, :])

		fs[t] = s_top.sum()/s_bottom.sum()
	
	score = -numpy.log(fs/h[p_internal]/numpy.sqrt(2*numpy.pi)).mean()
	
	if print_iterates:
		print(h, score)
	
	return score

def score_data_lwo_weave(h_sr, De, lwo_halfwidth = 0, print_iterates = False):
	h = h_sr*h_sr
	h_squared = h*h
	
	p_internal = (De.shape[2] - 1)

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled /= h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:p_internal].sum(2)
	S_top = S_bottom + De_scaled[:,:,p_internal]

	summands_bottom = numpy.exp(S_bottom)
	summands_top	= numpy.exp(S_top)

	N = De_scaled.shape[0]

	fs = numpy.empty(N, dtype = 'float32')

	# Pythonic implementation:
	# for t in range(N):
	# 	lb = numpy.max([0, t - lwo_halfwidth])
	# 	ub = numpy.min([N-1, t + lwo_halfwidth])

	# 	mask = numpy.array([1]*lb + [0]*(ub-lb+1) + [1]*(N - ub - 1),dtype=numpy.bool)

	# 	s_top = numpy.compress(mask, summands_top[t, :])
	# 	s_bottom = numpy.compress(mask, summands_bottom[t, :])

	# 	fs[t] = s_top.sum()/s_bottom.sum()

	# C-like implementation.

	code = """
	double acc_top;
	double acc_bottom;
	int lb;
	int ub;

	for(int i = 0; i < N; i++)
	{	
		acc_top = 0;
		acc_bottom = 0;
		if((i < lwo_halfwidth))
		{
			lb = 0;
		}
		else
		{
			lb = i - lwo_halfwidth;
		}

		if((i > N - 1 - lwo_halfwidth))
		{
			ub = N;
		}
		else
		{
			ub = i + lwo_halfwidth + 1;
		}

		for(int j = 0; j < lb; j++)
		{
			acc_top += summands_top(i, j);
		}

		for(int j = 0; j < lb; j++)
		{
			acc_bottom += summands_bottom(i, j);
		}

		for(int j = ub; j < N; j++)
		{
			acc_top += summands_top(i, j);
		}

		for(int j = ub; j < N; j++)
		{
			acc_bottom += summands_bottom(i, j);
		}

		fs(i) = acc_top / acc_bottom;
	}
	"""

	weave.inline(code,['summands_top', 'summands_bottom', 'N', 'lwo_halfwidth', 'fs'],
			   type_converters=weave.converters.blitz)


	score = -numpy.log(fs/h[p_internal]/numpy.sqrt(2*numpy.pi)).mean()
	
	if print_iterates:
		print(h, score)
	
	return score

def nll_data_lwo_weave(h_sr, De, lwo_halfwidth = 0):
	h = h_sr*h_sr
	h_squared = h*h
	
	p_internal = (De.shape[2] - 1)

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled /= h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:p_internal].sum(2)
	S_top = S_bottom + De_scaled[:,:,p_internal]

	summands_bottom = numpy.exp(S_bottom)
	summands_top	= numpy.exp(S_top)

	N = De_scaled.shape[0]

	fs = numpy.empty(N, dtype = 'float32')

	# Pythonic implementation:
	# for t in range(N):
	# 	lb = numpy.max([0, t - lwo_halfwidth])
	# 	ub = numpy.min([N-1, t + lwo_halfwidth])

	# 	mask = numpy.array([1]*lb + [0]*(ub-lb+1) + [1]*(N - ub - 1),dtype=numpy.bool)

	# 	s_top = numpy.compress(mask, summands_top[t, :])
	# 	s_bottom = numpy.compress(mask, summands_bottom[t, :])

	# 	fs[t] = s_top.sum()/s_bottom.sum()

	# C-like implementation.

	code = """
	double acc_top;
	double acc_bottom;
	int lb;
	int ub;

	for(int i = 0; i < N; i++)
	{	
		acc_top = 0;
		acc_bottom = 0;
		if((i < lwo_halfwidth))
		{
			lb = 0;
		}
		else
		{
			lb = i - lwo_halfwidth;
		}

		if((i > N - 1 - lwo_halfwidth))
		{
			ub = N;
		}
		else
		{
			ub = i + lwo_halfwidth + 1;
		}

		for(int j = 0; j < lb; j++)
		{
			acc_top += summands_top(i, j);
		}

		for(int j = 0; j < lb; j++)
		{
			acc_bottom += summands_bottom(i, j);
		}

		for(int j = ub; j < N; j++)
		{
			acc_top += summands_top(i, j);
		}

		for(int j = ub; j < N; j++)
		{
			acc_bottom += summands_bottom(i, j);
		}

		fs(i) = acc_top / acc_bottom;
	}
	"""

	weave.inline(code,['summands_top', 'summands_bottom', 'N', 'lwo_halfwidth', 'fs'],
			   type_converters=weave.converters.blitz)


	nlls = -numpy.log(fs/h[p_internal]/numpy.sqrt(2*numpy.pi))
	
	return nlls

def nll_data_lwo_bm(h_sr, De, lwo_halfwidth = 0):
	h = h_sr*h_sr
	h_squared = h*h
	
	p_internal = (De.shape[2] - 1)

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled /= h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:p_internal].sum(2)
	S_top = S_bottom + De_scaled[:,:,p_internal]

	summands_bottom = numpy.exp(S_bottom)
	summands_top	= numpy.exp(S_top)

	N = De_scaled.shape[0]

	fs = numpy.empty(N, dtype = 'float32')

	for t in range(N):
		lb = numpy.max([0, t - lwo_halfwidth])
		ub = numpy.min([N-1, t + lwo_halfwidth])

		mask = numpy.array([1]*lb + [0]*(ub-lb+1) + [1]*(N - ub - 1),dtype=numpy.bool)

		s_top = numpy.compress(mask, summands_top[t, :])
		s_bottom = numpy.compress(mask, summands_bottom[t, :])

		fs[t] = s_top.sum()/s_bottom.sum()
	
	nlls = -numpy.log(fs/h[p_internal]/numpy.sqrt(2*numpy.pi))
	
	return nlls

def score_data_weave(h_sr, De, print_iterates = False):
	h = h_sr*h_sr
	h_squared = h*h
	
	p_internal = (De.shape[2] - 1)

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled /= h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:p_internal].sum(2)
	S_top = S_bottom + De_scaled[:,:,p_internal]

	summands_bottom = numpy.exp(S_bottom)
	summands_top	= numpy.exp(S_top)

	N_eval = De_scaled.shape[0]
	N_train  = De_scaled.shape[1]

	fs = numpy.empty(N_eval, dtype = 'float32')

	# Pythonic implementation:
	# for t in range(N_eval):

	# 	s_top = summands_top[t, :]
	# 	s_bottom = summands_bottom[t, :]

	# 	fs[t] = s_top.sum()/s_bottom.sum()

	# C-like implementation.

	code = """
	double acc_top;
	double acc_bottom;

	for(int i = 0; i < N_eval; i++)
	{	
		acc_top = 0;
		acc_bottom = 0;

		for(int j = 0; j < N_train; j++)
		{
			acc_top += summands_top(i, j);
			acc_bottom += summands_bottom(i, j);
		}

		fs(i) = acc_top / acc_bottom;
	}
	"""

	weave.inline(code,['summands_top', 'summands_bottom', 'N_train', 'N_eval', 'fs'],
			   type_converters=weave.converters.blitz)


	score = -numpy.log(fs/h[p_internal]/numpy.sqrt(2*numpy.pi)).mean()
	
	if print_iterates:
		print(h, score)
	
	return score

def score_data_lwo_rawh_weave(h, De, lwo_halfwidth = 0):
	h_squared = h*h
	
	p_internal = (De.shape[2] - 1)

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled /= h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:p_internal].sum(2)
	S_top = S_bottom + De_scaled[:,:,p_internal]

	summands_bottom = numpy.exp(S_bottom)
	summands_top	= numpy.exp(S_top)

	N = De_scaled.shape[0]

	fs = numpy.empty(N, dtype = 'float32')

	# Pythonic implementation:
	# for t in range(N):
	# 	lb = numpy.max([0, t - lwo_halfwidth])
	# 	ub = numpy.min([N-1, t + lwo_halfwidth])

	# 	mask = numpy.array([1]*lb + [0]*(ub-lb+1) + [1]*(N - ub - 1),dtype=numpy.bool)

	# 	s_top = numpy.compress(mask, summands_top[t, :])
	# 	s_bottom = numpy.compress(mask, summands_bottom[t, :])

	# 	fs[t] = s_top.sum()/s_bottom.sum()

	# C-like implementation.

	code = """
	double acc_top;
	double acc_bottom;
	int lb;
	int ub;

	for(int i = 0; i < N; i++)
	{	
		acc_top = 0;
		acc_bottom = 0;
		if((i < lwo_halfwidth))
		{
			lb = 0;
		}
		else
		{
			lb = i - lwo_halfwidth;
		}

		if((i > N - 1 - lwo_halfwidth))
		{
			ub = N;
		}
		else
		{
			ub = i + lwo_halfwidth + 1;
		}

		for(int j = 0; j < lb; j++)
		{
			acc_top += summands_top(i, j);
		}

		for(int j = 0; j < lb; j++)
		{
			acc_bottom += summands_bottom(i, j);
		}

		for(int j = ub; j < N; j++)
		{
			acc_top += summands_top(i, j);
		}

		for(int j = ub; j < N; j++)
		{
			acc_bottom += summands_bottom(i, j);
		}

		fs(i) = acc_top / acc_bottom;
	}
	"""

	weave.inline(code,['summands_top', 'summands_bottom', 'N', 'lwo_halfwidth', 'fs'],
			   type_converters=weave.converters.blitz)


	score = -numpy.log(fs/h[p_internal]/numpy.sqrt(2*numpy.pi)).mean()
	
	return score

def score_data_lai(h_sr, De, lwo_halfwidth = 0):
	h = h_sr*h_sr
	h_squared = h*h
	
	p_internal = (De.shape[2] - 1)

	# We are now ready to *vary* h, which requires
	# a recompute for each value of h.

	# Rescale De by the bandwidths.

	De_scaled = De.copy()

	De_scaled /= h_squared

	# Each term of the KDE numerator and
	# denominator sum is given by summing
	# across the third dimension of De_scaled
	# and exponentiating.

	S_bottom = De_scaled[:,:,:p_internal].sum(2)
	S_top = S_bottom + De_scaled[:,:,p_internal]

	summands_bottom = numpy.exp(S_bottom)
	summands_top	= numpy.exp(S_top)

	N = De_scaled.shape[0]

	fs = numpy.empty(N, dtype = 'float32')

	# This is *wrong*, but I'm curious how much *faster* it is.

	s_top = summands_top.sum(1)
	s_bottom = summands_bottom.sum(1)

	fs = s_top.sum()/s_bottom.sum()
	
	score = -numpy.log(fs/h[p_internal]/numpy.sqrt(2*numpy.pi)).mean()
	
	return score

def integrand_flogf_er(x_future, X_futures, S_bottom, summands_bottom, h, h_squared, mask):
		future_term = -0.5*numpy.power(X_futures-x_future,2)/h_squared[-1]

		S_top = S_bottom+future_term

		summands_top = numpy.exp(S_top)
		
		N = len(summands_top)

		s_top = summands_top[mask]
		s_bottom = summands_bottom[mask]

		f = s_top.sum()/s_bottom.sum()/h[-1]/numpy.sqrt(2*numpy.pi)
		
		if f == 0:
			return 0
		else:
			return -f*numpy.log(f)

def eval_pred_dens_insample(x_future, X_futures, S_bottom, summands_bottom, h, h_squared, lwo_halfwidth, t):
		N = S_bottom.shape[0]

		lb = numpy.max([0, t - lwo_halfwidth])
		ub = numpy.min([N-1, t + lwo_halfwidth])

		mask = numpy.array([1]*lb + [0]*(ub-lb+1) + [1]*(N - ub - 1),dtype=numpy.bool)

		future_term = -0.5*numpy.power(X_futures-x_future,2)/h_squared[-1]

		S_top = S_bottom[t, :]+future_term

		summands_top = numpy.exp(S_top)

		s_top = summands_top[mask]
		s_bottom = summands_bottom[t, :][mask]

		# s_top = numpy.compress(mask, summands_top[t, :])
		# s_bottom = numpy.compress(mask, summands_bottom[t, :])

		f = s_top.sum()/s_bottom.sum()/h[-1]/numpy.sqrt(2*numpy.pi)
		
		return f

def moving_average(x, n=3):
	x = numpy.array(x)

	N = len(x)
	
	ma = numpy.zeros((N, 1))
	
	for t in range(N):	
		lb = numpy.max([0, t - n])
		ub = numpy.min([N-1, t + n])
	
		mask = numpy.array([0]*lb + [1]*(ub-lb+1) + [0]*(N - ub - 1),dtype=numpy.bool)
	
		ma[t] = numpy.nanmean(x[mask])
	
	return ma

def save_bandwidth_io(p_max, h, active_set, sd_x, sd_y, fname = None):
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Save out the bandwidths. Note that the bandwidths
	# are saved on the scale of the *original* data,
	# not on the scale of the std = 1 data.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if fname == None:
		fname = 'io'

	string_for_file = 'variable,lag,bandwidth\n'

	sorted_inds = active_set.argsort()

	for sorted_ind in sorted_inds:
		relative_index = active_set[sorted_ind]
		if relative_index < p_max:
			string_for_file += '{},{},{}\n'.format('y', p_max - relative_index, h[sorted_ind]*sd_y)
		else:
			string_for_file += '{},{},{}\n'.format('x', 2*p_max - relative_index, h[sorted_ind]*sd_x)

	with open('{}-io.bw'.format(fname), 'w') as wfile:
		wfile.write(string_for_file)

def save_bandwidth_o(p_max, h, active_set, sd_x, fname = None):
	if fname == None:
		fname = 'o'

	string_for_file = 'variable,lag,bandwidth\n'

	sorted_inds = active_set.argsort()

	for sorted_ind in sorted_inds:
		relative_index = active_set[sorted_ind]
		string_for_file += '{},{},{}\n'.format('x', p_max - relative_index, h[sorted_ind]*sd_x)

	with open('{}-o.bw'.format(fname), 'w') as wfile:
		wfile.write(string_for_file)

def load_bandwidth_io(fname, p_max):
	lag_y = []
	h_y   = []

	lag_x = []
	h_x   = []

	for line in islice(open(fname), 1, None):
		variable, lag_cur, h_cur = line.strip().split(',')
		
		if variable == 'y':
			lag_y += [int(lag_cur)]
			h_y   += [float(h_cur)]
		elif variable == 'x':
			lag_x += [int(lag_cur)]
			h_x   += [float(h_cur)]

	h = numpy.array(h_y + h_x)

	active_set = (p_max - numpy.array(lag_y)).tolist() + (2*p_max - numpy.array(lag_x)).tolist()

	return numpy.array(h), numpy.array(active_set)

def load_bandwidth_o(fname, p_max):
	lag_x = []
	h_x   = []

	for line in islice(open(fname + '-o.bw'), 1, None):
		variable, lag_cur, h_cur = line.strip().split(',')
		
		assert variable == 'x'

		lag_x += [int(lag_cur)]
		h_x   += [float(h_cur)]

	active_set = (p_max - numpy.array(lag_x)).tolist()

	return numpy.array(h_x), numpy.array(active_set)

def func_h_cutoff(h, rs, eps):
	return 1-numpy.exp(-0.5*rs/(h*h)) - numpy.sqrt(2*numpy.pi)*h*eps