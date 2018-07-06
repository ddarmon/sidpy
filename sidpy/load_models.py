import random
import math
import numpy
import matplotlib.pyplot as plt
import scipy.special
import os
import pandas

import sdeint

def load_model_data(model_name, N, ds_by = None):
	"""
	Generate time series from various model systems, including:
	
		arch
		(s)logistic
		(s)henon
		(s)lorenz
		(s)rossler
		(s)tent
		setar

	where the 's' prefix indicates a stochastic 
	difference / differential equation  version of the standard
	deterministic model.


	Parameters
	----------
	model_name : str
			The name of the model to use. See function description
			for list of available models.

	N : int
			The number of time points to simulate from the model
			system.

	ds_by : int
			Specify the amount to downsample a continuous-time
			time series by. By default, use the hard coded values.

	Returns
	-------
	x : numpy.array
			The time series.
	p_true : int
			The true model order of the system. If not
			a Markov model, p_true = numpy.inf
	model_type : str
			The model type.

	"""

	if model_name == 'arch':
		# Simulate from an ARCH(1) model:
		model_type = 'arch'
		p_true = 1
		x = numpy.zeros(N)
		u = numpy.random.randn(N)

		x[:1] = u[:1]

		for n in range(1, x.shape[0]):
			x[n] = numpy.sqrt(0.5 + 0.5*x[n-1]**2)*u[n]

		# Simulate from an ARCH(2) model:
		# p_true = 2

		# x = numpy.zeros(N)
		# u = numpy.random.randn(N)

		# x[:2] = u[:2]

		# for n in range(2, x.shape[0]):
		# 	x[n] = numpy.sqrt(0.5 + 0.5*x[n-1]**2 + 0.5*x[n-2]**2)*u[n]

	if model_name == 'slogistic' or model_name == 'logistic':
		# NLAR(1) model from:
		# 
		# **Estimation of conditional densities and sensitivity measures in nonlinear dynamical systems**
		model_type = 'nlar'
		p_true = 1
		x = numpy.zeros(N)

		if model_name == 'slogistic':
			noise = (numpy.random.rand(N, 48) - 0.5)/2./16.
			noise = noise.sum(1)
		else:
			noise = numpy.zeros(N)

		x[0] = numpy.random.rand(1)
		for t in range(1, x.shape[0]):
			x[t] = 3.68*x[t-1]*(1 - x[t-1]) + 0.4*noise[t]

			if x[t] <= 0 or x[t] >= 1:
				x[t] = numpy.random.rand(1)

	if model_name == 'shenon' or model_name == 'henon':
		# NLAR(2) model from:
		# 
		# **On prediction and chaos in stochastic systems**
		#
		# Namely, stochastic Henon.
		#
		# Also on page 188 (dp 200) of *Chaos: A Statistical Perpsective*.
		model_type = 'nlar'
		p_true = 2

		burnin = 1000

		x = numpy.zeros(N + burnin)

		if model_name == 'shenon':
			# From original paper:
			noise = (numpy.random.rand(N + burnin, 48) - 0.5)/2.
			noise = noise.sum(1)
		else:
			noise = numpy.zeros(N + burnin)

		x[:2] = 10*numpy.random.rand(1) - 5
		for t in range(2, x.shape[0]):
			x[t] = 6.8 - 0.19*x[t-1]**2 + 0.28*x[t-2] + 0.2*noise[t]

			if x[t] < -10:
				x[t] = 10*numpy.random.rand(1) - 5

		x_w_burnin = x.copy()
		x = x[burnin:]

	if model_name == 'stent' or model_name == 'tent':
		# Tent map with noise as from the logistic map.
		# This is also a SETAR(1; 1, 1) model, in the
		# vein of Tong.

		model_type = 'nlar'
		p_true = 1

		x = numpy.zeros(N)

		if model_name == 'stent':
			noise = (numpy.random.rand(N, 48) - 0.5)/2./16.
			noise = noise.sum(1)
		else:
			noise = numpy.zeros(N)

		# No noise:
		# noise = numpy.zeros(N)

		x[0] = numpy.random.rand(1)
		for t in range(1, x.shape[0]):
		    if x[t-1] < 0.5:
		        x[t] = 3.68/2.*x[t-1] + 0.4*noise[t]
		    else:
		        x[t] = 3.68/2. - 3.68/2.*x[t-1] + 0.4*noise[t]

		    if x[t] <= 0:
		       x[t] = numpy.random.rand(1)

	if model_name == 'setar':
		model_type = 'setar'

		# Simulate from a SETAR(2; 1, 1) model:

		# p_true = 1

		# x = numpy.zeros(N)
		# u = numpy.random.randn(N)

		# x[:1] = u[:1]

		# for n in range(1, x.shape[0]):
		# 	# From "Sieve bootstrap for time series" by Buhlmann in Bernoulli
		# 	if x[n-1] <= 0:
		# 		x[n] = 1.5 - 0.9*x[n-1] + u[n]
		# 	else:
		# 		x[n] = -0.4 - 0.6*x[n-1] + u[n]

			# if x[n-1] <= 0:
			# 	x[n] = 1.5 - 0.9*x[n-1] + 0.5*u[n]
			# else:
			# 	x[n] = -0.4 - 0.6*x[n-1] + u[n]

			# if x[n-1] <= 0:
			# 	x[n] = 1.5 - 0.9*x[n-1] + 0.5*u[n]
			# else:
			# 	x[n] = -0.4 + 0.1*x[n-1] + u[n]

		# Simulate from a SETAR(2; 2, 2) model:
		# p_true = 2
		# x = numpy.zeros(N)
		# u = numpy.random.randn(N)

		# x[:2] = u[:2]

		# for n in range(2, x.shape[0]):
		# 	if x[n-1] <= 0:
		# 		x[n] = 1.5 - 0.9*x[n-1] + 0.2*x[n-2] + u[n]
		# 	else:
		# 		x[n] = -0.4 - 0.6*x[n-1] + 0.1*x[n-2] + u[n]

		# Simulate from a SETAR(2; 2, 2) model used to model 
		# the Canadian Lynx data from Tong's *Non-linear 
		# Time Series*, p. 178:
		p_true = 2
		x = numpy.zeros(N)
		u = numpy.random.randn(N)

		x[:2] = 2*numpy.random.rand(2) + 2

		for n in range(2, x.shape[0]):
		  if x[n-2] <= 3.25:
		      x[n] = 0.62 + 1.25*x[n-1] - 0.43*x[n-2] + numpy.sqrt(0.0381)*u[n]
		  else:
		      x[n] = 2.25 + 1.52*x[n-1] - 1.24*x[n-2] + numpy.sqrt(0.0626)*u[n]

	if 'lorenz' in model_name:
		model_type = 'nlar'

		p_true = 4

		h = 0.05

		if ds_by == None:
			ds_by = 2

		ttot = N*h*ds_by
		tburn = 20
		tf = ttot + tburn

		tspan = numpy.linspace(0.0, tf, int(tf/h))
		x0 = numpy.array([1.0, 1.0, 1.0])

		params = [10., 28., 8./3.] # The parameters [s, r, b] for the canonical Lorenz equations

		def F(X, t):
			s = params[0]
			r = params[1]
			b = params[2]

			x = X[0]
			y = X[1]
			z = X[2]

			dX = numpy.array([s*(y - x), x*(r - z) - y, x*y - b*z])

			return dX

		def G(x, t):
		    return B

		dim = 3

		if model_name == 'slorenz':
			dyn_noise = 2
		else:
			dyn_noise = 0

		B = numpy.diag([dyn_noise]*dim)

		result = sdeint.itoint(F, G, x0, tspan)

		tspan = tspan[int(tburn/h)::ds_by]
		result = result[int(tburn/h)::ds_by, :]

		x = result[:, 0]

	if 'rossler' in model_name:
		model_type = 'nlar'

		p_true = 4

		h = 0.05

		if ds_by == None:
			ds_by = 5

		ttot = N*h*ds_by
		tburn = 100
		tf = ttot + tburn

		tspan = numpy.linspace(0.0, tf, int(tf/h))
		x0 = numpy.array([1.0, 1.0, 1.0])

		params = [0.1, 0.1, 14]

		def F(X, t):
			a = params[0]
			b = params[1]
			c = params[2]

			x = X[0]
			y = X[1]
			z = X[2]

			dX = numpy.array([-y - z, x + a*y, b + z*(x - c)])

			return dX

		def G(x, t):
		    return B

		dim = 3

		if model_name == 'srossler':
			dyn_noise = 0.1
		else:
			dyn_noise = 0.0

		B = numpy.diag([dyn_noise]*dim)
		B[2, 2] = 0

		result = sdeint.itoint(F, G, x0, tspan)

		tspan = tspan[int(tburn/h)::ds_by]
		result = result[int(tburn/h)::ds_by, :]

		x = result[:, 0]
		y = result[:, 1]
		z = result[:, 2]

	if model_name == 'snanopore':
		a=1.
		b=1.
		c=1.
		km=5.
		kp=1.
		gamx=1.
		gamy=100.
		alpha=0.1
		beta=0.1

		model_type = 'nlar'

		p_true = numpy.inf

		if ds_by == None:
			ds_by = 2

		delta_t=0.25
		T_final = ds_by*N*delta_t

		M = int(T_final/delta_t + 1)

		tspan = numpy.linspace(0.0, T_final, M)
		x0 = numpy.array([1.0, -0.5])

		def f(z, t):
		#z=[x,y]
			dx=-(1/gamx)*(a*numpy.power(z[0],3)-b*z[0]+c*z[1])
			if z[0]<0:
				H=0
			else:
				H=1
			if z[0]>0:
				Hm=0
			else:
				Hm=1
			dy=(1/gamy)*(kp*H-km*Hm)
			return numpy.array([dx,dy]) 

		def G(z, t):
			B=numpy.diag([alpha,beta])
			return B

		print 'Solving SDE...'

		result = sdeint.itoint(f, G, x0, tspan)

		x = result[::ds_by, 0]

	if model_name == 'shadow_crash':
		p_true = 4
		model_type = 'nlar'

		h = 0.05

		if ds_by == None:
			ds_by = 2

		ttot = N*h*ds_by
		tburn = 20
		tf = ttot + tburn

		tspan = numpy.linspace(0.0, tf, int(tf/h))
		x0 = numpy.array([1.0, 1.0])

		def F(X, t):
			b = 0.42
			g = -0.04
			
			x = X[0]
			z = X[1]
			
			dX = numpy.array([x - x**2*numpy.exp(-b*x*z), z - z**2*numpy.exp(-g*x)])

			return dX

		def G(X, t):
			# B = numpy.diag([0.4, 0.01])
			B = numpy.diag([0.2, 0.01])

			x = X[0]
			z = X[1]

			B[0, 0] = x*B[0, 0]
			B[1, 1] = z*B[1, 1]
			
			return B

		dim = 2

		result = sdeint.itoint(F, G, x0, tspan)

		tspan = tspan[int(tburn/h)::ds_by]
		result = result[int(tburn/h)::ds_by, :]

		x = result[:, 0]

	return x, p_true, model_type