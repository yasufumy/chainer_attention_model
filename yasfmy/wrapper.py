from functools import wraps

from chainer import cuda
import numpy as np

class xpmeta(type):
    def __new__(cls, clsname, bases, clsdict):
        try:
            cuda.check_cuda_available()
            xp = cuda.cupy
        except RuntimeError:
            xp = np
        return xp

class xp(metaclass=xpmeta):
    pass
