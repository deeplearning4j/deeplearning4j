import unittest

import jumpy as jp
import numpy as np


import gc
gc.disable()

class TestArrayCreation(unittest.TestCase):
    init()

    def test_arr_creation(self):
        x_np = np.random.random((100, 32, 16))
        x_jp = jp.array(x_np)
        x_np_2 = x_jp.numpy()
        self.assertEquals(x_np.shape, x_jp.shape)
        self.assertEquals(x_np.shape, x_np_2.shape)
        x_np = x_np.ravel()
        x_np_2 = x_np_2.ravel()
        assertEquals(list(x_np), list(x_np_2))
