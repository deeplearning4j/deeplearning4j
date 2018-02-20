import unittest

from jumpy import *
import numpy as np
from builtins import range

class TestBufferCreation(unittest.TestCase):
    init()

    def test_buffer_creation(self):
        sizes = [1, 4, 32, 100, 128, 1000, 1024]
        for size in sizes:
            buffer = get_buffer_from_arr(np.linspace(1, size, size))
            self.assertEqual(size, buffer.length())
            self.assertEqual(8, buffer.element_size())
            for i in range(0, size):
                self.assertEqual(i + 1, buffer[i])

    def test_buffer_creation_float(self):
        sizes = [1, 4, 32, 100, 128, 1000, 1024]
        for size in sizes:
            arr = np.asarray(np.linspace(1, size, size), dtype=np.float32)
            buffer = get_buffer_from_arr(arr)
            self.assertEquals(size, buffer.length())
            self.assertEqual(4, buffer.element_size())
            for i in range(0, size):
                self.assertEqual(i + 1, buffer[i])
