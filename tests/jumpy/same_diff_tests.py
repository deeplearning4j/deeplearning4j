from jumpy import same_diff_create
import unittest
import jumpy
import numpy as np

class TestLogisticRegression(unittest.TestCase):
    def test_logistic(self):
        jumpy.init()
        same_diff = same_diff_create()
        a = np.linspace(1,4,4)
        nd4j_arr = jumpy.from_np(a).array
        test_var = same_diff.var('x',nd4j_arr)
        sigmoided_var = same_diff.sigmoid(test_var)
