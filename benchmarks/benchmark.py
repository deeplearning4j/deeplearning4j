import jumpy
from jumpy import Nd4jArray
import numpy as np
from random import randint
import timeit
import gc

gc.disable()
jumpy.disable_gc()


class Benchmark(object):
    def __init__(self, n=100):
        print 'Running tests with [',n,'x',n,'] dimensionality'
        self.n = n
        self.m = 200;
        self.np_arr = []
        self.nd4j_arr = []
        for counter in range(0, self.m + 1):
            self.np_arr.append(np.linspace(1, n * n, n * n).reshape((n, n)))

        for counter in range(0, self.m + 1):
            self.nd4j_arr.append(jumpy.from_np(self.np_arr[counter]))
        # self.nd4j_arr = jumpy.from_np(np.copy(self.np_arr))

    def run_nd4j_scalar(self):
        self.nd4j_arr[randint(0, self.m)] += 1.0172

    def run_numpy_scalar(self):
        self.np_arr[randint(0, self.m)] += 1.0172

    def run_nd4j_add(self):
        self.nd4j_arr[randint(0, self.m)] += self.nd4j_arr[randint(0, self.m)]

    def run_numpy_add(self):
        self.np_arr[randint(0, self.m)] += self.np_arr[randint(0, self.m)]

    def run_numpy_sub(self):
        self.np_arr[randint(0, self.m)] -= self.np_arr[randint(0, self.m)]

    def run_nd4j_sub(self):
        self.nd4j_arr[randint(0, self.m)] -= self.nd4j_arr[randint(0, self.m)]

    def run_nd4j_mmul(self):
        jumpy.dot(self.nd4j_arr[randint(0, self.m)], self.nd4j_arr[randint(0, self.m)])

    def run_numpy_mmul(self):
        np.dot(self.np_arr[randint(0, self.m)], self.np_arr[randint(0, self.m)])

    def run_benchmark(self, n_trials=1000):
        print 'nd4j scalar ', timeit.timeit(self.run_nd4j_scalar, number=n_trials)
        print 'numpy scalar ', timeit.timeit(self.run_numpy_scalar, number=n_trials)
        print 'nd4j add ', timeit.timeit(self.run_nd4j_add, number=n_trials)
        print 'numpy add ', timeit.timeit(self.run_numpy_add, number=n_trials)
        print 'nd4j sub ', timeit.timeit(self.run_nd4j_sub, number=n_trials)
        print 'numpy sub ', timeit.timeit(self.run_numpy_sub, number=n_trials)
        print 'nd4j mmul ', timeit.timeit(self.run_nd4j_mmul, number=n_trials)
        print 'numpy mmul ', timeit.timeit(self.run_numpy_mmul, number=n_trials)


benchmark = Benchmark()
benchmark.run_benchmark()
