from sidpy import *

x = numpy.arange(100).reshape(1, -1)
y = numpy.arange(100).reshape(1, -1)

q = 5
p = 2
delay = 0

Z = stack_io(y, x, q, p, delay)

q = 5
p = 6
delay = 2

Z = stack_io(y, x, q, p, delay)

print(Z[:10, :])

print('')

print(Z[-10:, :])

q = 11
p = 10
delay = -2

Z = stack_io(y, x, q, p, delay)

print(Z[:10, :])

print('')

print(Z[-10:, :])