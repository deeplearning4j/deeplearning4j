"""
Shape:
 50  27
Stride:
 675  9
Order f
"""


import numpy as np
a = np.random.random((2,5,5,3,3,3))
print([(stride / 8) for stride in a.strides])

b = np.reshape(a,(50,27),'f')
print([(stride / 8) for stride in b.strides])
