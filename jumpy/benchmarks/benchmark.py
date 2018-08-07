################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################


import jumpy as jp
import numpy as np
from random import randint
import timeit
import gc

gc.disable()
jp.disable_gc()


class Benchmark(object):
    def __init__(self, n=1000):
        print 'Running tests with [',n,'x',n,'] dimensionality'
        self.n = n
        self.m = 200
        self.np_arr = []
        self.nd4j_arr = []
        for counter in range(0, self.m + 1):
            self.np_arr.append(np.linspace(1, n * n, n * n).reshape((n, n)))

        for counter in range(0, self.m + 1):
            self.nd4j_arr.append(jp.array(self.np_arr[counter]))

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
        jp.dot(self.nd4j_arr[randint(0, self.m)], self.nd4j_arr[randint(0, self.m)])

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
