import itertools
import numpy

n = 14

my_set_x = numpy.arange(1, n+1)
my_set_y = numpy.arange(1, n+1)

print(len(my_set_x))
print(len(my_set_y))

subsets_x=[(my_set_x[i], my_set_x[j]) for i in range(n-1) for j in range(i+1,n)] 
#print(subsets_x)
print(len(subsets_x))

subsets_y=[(my_set_y[i], my_set_y[j]) for i in range(n-1) for j in range(i+1,n)] 
#print(subsets_y)
print(len(subsets_y))

universe_rects = []

import math

for r in itertools.product(subsets_x, subsets_y): 
    if numpy.abs(r[0][1]-r[0][0]) != numpy.abs(r[1][1]-r[1][0]):
        universe_rects.append(r)

print(len(universe_rects))
print(len(universe_rects)/n)

# #output
# 14
# 14
# 91
# 91
# 7462
# 533.0