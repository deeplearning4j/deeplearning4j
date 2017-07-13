import jumpy
from jumpy import Nd4jArray
import numpy as np
import timeit
import gc
gc.disable()

jumpy.disable_gc()

class Benchmark(object):
    def __init__(self, n=80):
        self.np_arr = np.linspace(1, n * n, n * n).reshape((n, n))
        self.nd4j_arr = Nd4jArray(jumpy.nd4j.linspace(1,n * n,n * n).reshape(n,n))
        #self.nd4j_arr = jumpy.from_np(np.copy(self.np_arr))

    def run_nd4j_add(self):
        self.nd4j_arr += self.nd4j_arr

    def run_numpy_add(self):
        self.np_arr += self.np_arr

    def run_numpy_sub(self):
        self.np_arr -= self.np_arr

    def run_nd4j_sub(self):
        self.nd4j_arr -= self.nd4j_arr

    def run_nd4j_mmul(self):
        jumpy.dot(self.nd4j_arr,self.nd4j_arr)

    def run_numpy_mmul(self):
        np.dot(self.np_arr,self.np_arr)

    def run_benchmark(self,n_trials=1000):
        print 'nd4j add ', timeit.timeit(self.run_nd4j_add, number=n_trials)
        print 'numpy add ', timeit.timeit(self.run_numpy_add, number=n_trials)
        print 'nd4j sub ', timeit.timeit(self.run_nd4j_sub, number=n_trials)
        print 'numpy sub ', timeit.timeit(self.run_numpy_sub, number=n_trials)
        print 'nd4j mmul ', timeit.timeit(self.run_nd4j_mmul, number=n_trials)
        print 'numpy mmul ', timeit.timeit(self.run_numpy_mmul, number=n_trials)



benchmark = Benchmark()
benchmark.run_benchmark()



