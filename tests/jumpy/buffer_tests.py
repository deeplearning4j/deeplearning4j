import unittest

from jumpy import *
import numpy as np
from builtins import range

class TestBufferCreation(unittest.TestCase):
    init()

    def test_buffer_creation(self):
        buffer = get_buffer_from_arr(np.linspace(1, 4, 4))
        self.assertEqual(4, buffer.length())
        self.assertEqual(8, buffer.element_size())
        for i in range(0, 4):
            self.assertEqual(i + 1, buffer[i])

    def test_buffer_creation_float(self):
        arr = np.asarray(np.linspace(1, 4, 4), dtype=np.float32)
        buffer = get_buffer_from_arr(arr)
        self.assertEquals(4, buffer.length())
        self.assertEqual(4, buffer.element_size())
        for i in range(0, 4):
            self.assertEqual(i + 1, buffer[i])
