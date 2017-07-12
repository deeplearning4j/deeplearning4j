import unittest


from jumpy import *

class TestArrayCreation(unittest.TestCase):
    def test_arr_creation(self):
        set_data_type('double')
        a = np.linspace(1, 4, 4).reshape(2, 2)
        nd4j_arr = from_np(a)
        length = nd4j_arr.length()
        self.assertEqual(4, length)
        self.assertEquals(list(a.shape), nd4j_arr.shape())
        self.assertEquals(map(lambda x: x / 8, list(a.strides)), nd4j_arr.stride())
        self.assertEquals(1.0, nd4j_arr.data().getDouble(0))
        #self.assertEquals(1.0, nd4j_arr.getDouble(0, 0))

    def test_arr_creation_vector(self):
        a = np.linspace(1, 4, 4)
        nd4j_arr = from_np(a)
        self.assertEqual(2, nd4j_arr.rank())

    def test_add(self):
        a = np.reshape(np.linspace(1, 4, 4),(2,2))
        nd4j_arr = from_np(a)
        nd4j_arr_output = nd4j_arr + nd4j_arr
        nd4j_times_2 = nd4j_arr * 2.0

