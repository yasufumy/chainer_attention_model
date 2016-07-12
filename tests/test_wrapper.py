import prepare
import unittest
from wrapper import xp
from chainer import cuda
from chainer import Variable

class WrapperTestCase(unittest.TestCase):

    def test_xp(self):
        try:
            cuda.check_cuda_available()
            module = 'cupy'
        except:
            module = 'numpy'
        self.assertEqual(xp.__name__, module)

    def test_Zeros(self):
        zeros = xp.Zeros((1, 1), dtype=xp.float32)
        self.assertEqual(type(zeros), Variable)
        self.assertEqual(zeros.data[0][0], 0.0)
        self.assertEqual(zeros.data.dtype, xp.float32)

    def test_Array(self):
        arr = xp.Array([0], dtype=xp.int32)
        self.assertEqual(type(arr), Variable)
        self.assertEqual(arr.data[0], 0)
        self.assertEqual(arr.data.dtype, xp.int32)
