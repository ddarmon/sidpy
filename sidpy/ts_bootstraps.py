import numpy
import sidpy

def circular_block_bootstrap_multvar(x, p_max, block_length = None, num_blocks = None, as_data_matrix = True):
	T = x.shape[1]

	if block_length == None:
		block_length = int(T**(1./3))

	if num_blocks == None:
		num_blocks = int(numpy.ceil((T - (p_max + 1))/block_length))

	# Create embedding matrix, with wrapping around for the circular
	# bootstrap.

	X = sidpy.embed_ts_multvar(x, p_max, circular = True)

	# Allocate an empty matrix for the circular block bootstrap replicate.

	X_boot = numpy.zeros((X.shape[0], num_blocks*block_length, X.shape[2]))

	# Populate X_boot by randomly sampling blocks of length block_length
	# from X

	block_template = numpy.arange(block_length)

	offsets = numpy.random.randint(X.shape[1], size = num_blocks)

	for i, random_offset in enumerate(offsets):
		X_boot[:, i*block_length:(i + 1)*block_length, :] = X[:, (block_template + random_offset) % X.shape[1], :]

	# Truncate extra rows to make X_boot the same length as a standard
	# embedding matrix.

	X_boot = X_boot[:, :(T - p_max), :]

	if as_data_matrix:
		return X_boot, block_length, num_blocks
	else:
		x_boot = numpy.concatenate((X_boot[:, :, 0], X_boot[:, -1, [1]]), axis = 1)

		return x_boot, block_length, num_blocks