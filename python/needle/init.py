import math
import needle as ndl


def uniform(x, low=0.0, high=1.0):
    x.data = ndl.randu(x.shape, low=low, high=high, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    return x

def normal(x, mean=0.0, std=1.0):
    x.data = ndl.randn(x.shape, mean=mean, std=std, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    return x

def constant(x, c=0.0):
    x.data = ndl.full(x.shape, c, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    return x

def ones(x):
    x.data = ndl.ones_like(x, device=x.device, requires_grad=x.requires_grad)
    return x

def zeros(x):
    x.data = ndl.zeros_like(x, device=x.device, requires_grad=x.requires_grad)
    return x

def _calculate_fans(x, mode=None):
    if mode == "fan_out":
        return x.shape[-1]
    elif mode == "fan_in":
        return x.shape[-2]
    elif mode is None:
        return x.shape[-1] + x.shape[-2]


def xavier_uniform(x, gain=1.0):
    a = gain * math.sqrt(6 / _calculate_fans(x))
    return uniform(x, -a, a)


def xavier_normal(x, gain=1.0):
    a = gain * math.sqrt(2 / _calculate_fans(x))
    return normal(x, 0, a)


def kaiming_uniform(x, mode='fan_in', nonlinearity='relu'):
    if nonlinearity == 'relu':
        gain = math.sqrt(2)
    else:
        gain = 1.0

    a = gain * math.sqrt(3 / _calculate_fans(x, mode))
    return uniform(x, -a, a)


def kaiming_normal(x, mode='fan_in', nonlinearity='relu'):
    if nonlinearity == 'relu':
        gain = math.sqrt(2)
    else:
        gain = 1.0

    a = gain / math.sqrt(_calculate_fans(x, mode))
    return normal(x, 0, a)
