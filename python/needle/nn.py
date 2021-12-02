"""The module.
"""
from __future__ import annotations
from typing import List

import needle as ndl
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters.
    """

def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []

def _child_modules(value: object) -> List[Module]:
    if isinstance(value, Module):
        return [value] + _child_modules(value.__dict__)
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _child_modules(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _child_modules(v)
        return params
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module.
        """
        return _unpack_params(self.__dict__)

    def _children(self) -> List[Module]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = np.sqrt(1 / in_features)
        self.weight = Parameter(ndl.randu((in_features, out_features), low=-k, high=k,
                                device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(ndl.randu((out_features,), low=-k, high=k,
                              device=device, dtype=dtype, requires_grad=True)) if bias else None

    def forward(self, x: Tensor)-> Tensor:
        N = x.shape[:-1]
        x = x.reshape(N + (1, self.in_features))
        out = x @ self.weight.broadcast_to(N + (self.in_features, self.out_features))
        if self.bias is not None:
            out += self.bias.broadcast_to(N + (1, self.out_features))
        return out.reshape(N + (self.out_features,))


class ReLU(Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ndl.relu(x)


class Sequential(Module):
    def __init__(self, *modules, device=None, dtype="float32"):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.modules:
            x = mod(x)
        return x


class SoftmaxLoss(Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor):
        out = ndl.logsoftmax(x) * ndl.one_hot(y, num_classes=x.shape[-1], dtype=x.dtype, device=x.device)
        return -out.sum() / x.shape[0]


class BatchNorm(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(ndl.ones(self.dim, requires_grad=True))
        self.bias = Parameter(ndl.zeros(self.dim, requires_grad=True))
        self.running_mean = None
        self.running_var = None

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose((1, -1))

        stats_dims = tuple(list(range(len(x.shape) - 1)))
        N = np.prod([x.shape[d] for d in stats_dims])

        x_mean = x.sum(axes=stats_dims) / N
        x_var = ((x - x_mean.broadcast_to(x.shape)) ** 2).sum(axes=stats_dims) / N

        if self.running_mean is None:
            self.running_mean = ndl.zeros_like(x_mean, device=x_mean.device)
            self.running_var = ndl.ones_like(x_var, device=x_var.device)

        if self.training:
            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * x_mean).detach()
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * x_var * (N / (N - 1))).detach()
            mu, sig = x_mean, x_var
        else:
            mu, sig = self.running_mean, self.running_var

        weighted_normalized = (x - mu.broadcast_to(x.shape)) * self.weight.broadcast_to(x.shape) / ((sig + self.eps) ** 0.5).broadcast_to(x.shape)
        return (weighted_normalized + self.bias.broadcast_to(x.shape)).transpose((1, -1))


class LayerNorm(Module):
    def __init__(self, dims, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        self.weight = Parameter(ndl.ones(self.dims, requires_grad=True))
        self.bias = Parameter(ndl.zeros(self.dims, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        feature_dims = tuple(list(range(len(x.shape) - len(self.dims), len(x.shape))))
        x_demean = x - x.mean(axes=feature_dims, keepdims=True).broadcast_to(x.shape)
        x_var = (x_demean ** 2).mean(axes=feature_dims, keepdims=True).broadcast_to(x.shape)
        w, b = self.weight.broadcast_to(x.shape), self.bias.broadcast_to(x.shape)
        return x_demean / ((x_var + self.eps) ** 0.5) * w + b


class Dropout(Module):
    def __init__(self, drop_prob, device=None, dtype="float32"):
        super().__init__()
        self.p = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x * ndl.randb(x.shape, n=1, p=(1 - self.p), device=x.device) / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module, device=None, dtype="float32"):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Identity(Module):
    def __init__(self, *args, device=None, dtype="float32", **kwargs):
        super().__init__()

    def forward(self, x):
        return x
