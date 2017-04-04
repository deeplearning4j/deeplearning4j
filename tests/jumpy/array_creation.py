import unittest

import numpy as np
from jnius import JavaException

from jumpy import *


class TestArrayCreation(unittest.TestCase):
    def test_arr_creation(self):
        a = np.linspace(1, 4, 4).reshape(2, 2)
        nd4j_arr = from_np(a)
        length = nd4j_arr.length()
        self.assertEqual(4, length)
        self.assertEquals(list(a.shape),nd4j_arr.shape())
        self.assertEquals(map(lambda x: x / 8, list(a.strides)), nd4j_arr.stride())
        try:
            self.assertEquals(1.0, nd4j_arr.data().getDouble(0))
            self.assertEquals(1.0, nd4j_arr.getDouble(0, 0))
            system.out.println(nd4j_arr)
        except JavaException as e:
            print e.stacktrace

