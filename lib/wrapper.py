from chainer import cuda
from chainer import Variable as V
import numpy as np

def generate_variabled_method(xp, name):
    def variabled(*args, volatile='OFF', **kwargs):
        return V(xp.__dict__[name](*args, **kwargs), volatile=volatile)
    return variabled

class xpmeta(type):
    def __new__(cls, clsname, bases, clsdict):
        try:
            cuda.check_cuda_available()
            xp = cuda.cupy
        except RuntimeError:
            xp = np
        methods = ['array', 'zeros']
        for name in methods:
            setattr(xp, name.capitalize(), generate_variabled_method(xp, name))
        setattr(xp, 'Variable', V)
        return xp

class xp(metaclass=xpmeta):
    pass
