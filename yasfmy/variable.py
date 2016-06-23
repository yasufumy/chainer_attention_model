from functools import wraps

from chainer import Variable as V
from chainer import cuda
import numpy as np

wrap_list = ['array', 'zeros', 'asarray']

def wrap_variable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return V(result)
    return wrapper

def wrap_method_with_variable(cls):
    for name in wrap_list:
        setattr(cls, name, wrap_variable(cls.__dict__[name]))
    return cls

class VariableMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        try:
            cuda.check_cuda_available()
            xp = cuda.cupy
        except:
            xp = np
        return wrap_method_with_variable(xp)

class Variable(metaclass=VariableMeta):
    pass
