import jumpy
import numpy as np
import timeit


class Benchmark(object):
    def __init__(self):
        self.np_arr = np.linspace(1, 10000, 10000)
        self.nd4j_arr = jumpy.from_np(self.np_arr)

    def run_nd4j(self):
        self.nd4j_arr += self.nd4j_arr

    def run_numpy(self):
        self.np_arr += self.np_arr

    def run_benchmark(self,n_trials=1000):
        print 'nd4j ', timeit.timeit(self.run_nd4j, number=n_trials)
        print 'numpy ', timeit.timeit(self.run_numpy, number=n_trials)



benchmark = Benchmark()
benchmark.run_benchmark()



