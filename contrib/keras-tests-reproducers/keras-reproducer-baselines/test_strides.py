import numpy as np
a = np.ones((3,3,3,3))
print([stride /8 for stride in a.strides])