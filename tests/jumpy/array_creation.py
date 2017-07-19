import unittest

from jumpy import *

import gc
gc.disable()

class TestArrayCreation(unittest.TestCase):
    init()

    def test_arr_creation(self):
        a = np.linspace(1, 4, 4).reshape(2, 2)
        nd4j_arr = from_np(a)
        length = nd4j_arr.length()
        self.assertEqual(4, length)
        self.assertEquals(list(a.shape), nd4j_arr.shape())
        strides_assertion = [2,1]
        self.assertEquals(strides_assertion, nd4j_arr.stride())
        for i in xrange(0, 4):
            self.assertEquals(i + 1, nd4j_arr.data().getDouble(i))

        self.assertEquals(1.0, nd4j_arr[0])

    def test_arr_creation_vector(self):
        a = np.linspace(1, 4, 4).reshape((1, 4))
        nd4j_arr = from_np(a)
        self.assertEqual(2, nd4j_arr.rank())
        self.assertEquals(list(a.shape), nd4j_arr.array.shape())


    def test_add(self):
        a = np.reshape(np.linspace(1, 4, 4), (2, 2))
        nd4j_arr = from_np(a)
        nd4j_arr_output = nd4j_arr + nd4j_arr
        nd4j_times_2 = nd4j_arr * 2.0
