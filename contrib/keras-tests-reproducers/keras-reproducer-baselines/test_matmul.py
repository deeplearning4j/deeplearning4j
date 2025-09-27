import numpy as np
a = np.random.rand(49,2,20)
b = np.random.rand(49,20,5)
c = np. matmul(a,b)
print(c.shape)