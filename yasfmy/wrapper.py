from functools import wraps

from chainer import cuda
from chainer import Variable
import numpy as np

def wrap_variable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return Variable(result)

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
