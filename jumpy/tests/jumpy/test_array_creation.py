import unittest

import jumpy as jp
import numpy as np


import gc
gc.disable()


class TestArrayCreation(unittest.TestCase):

    def setUp(self):
        self.x_np = np.random.random((100, 32, 16))
        self.x_jp = jp.array(self.x_np)
        self.x_np_2 = x_jp.numpy()

    def test_arr_creation(self):
        self.assertEquals(self.x_np.shape, self.x_jp.shape)
        self.assertEquals(self.x_np.shape, self.x_np_2.shape)
        x_np = self.x_np.ravel()
        x_np_2 = x_np_2.ravel()
        assertEquals(list(x_np), list(x_np_2))
