"""Optimization module"""
from typing import List
import needle as ndl
import numpy as np

from needle.autograd import Tensor


class Optimizer:
    def __init__(self, params: List[ndl.Tensor]):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.delta = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            u = self.delta.get(p, ndl.zeros_like(grad, device=grad.device))
            self.delta[p] = (self.momentum * u + grad).data
            self._set_val(p, p - self.lr * self.delta[p])

    def _set_val(self, tensor, value):
        tensor.data = Tensor(value, device=tensor.device, dtype=tensor.dtype)


class Adam(Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            grad = (p.grad + self.weight_decay * p).detach()
            m_ = self._m_update(self.m, p, grad, self.beta1).detach()
            v_ = self._m_update(self.v, p, grad ** 2, self.beta2).detach()
            self._set_val(p, p - self.lr * m_ / (v_ ** 0.5 + self.eps))

    def _m_update(self, group: dict, key, new, beta):
        old = group.get(key, ndl.zeros_like(new, device=new.device))
        out = group[key] = (beta * old + (1 - beta) * new).data
        return (out / (1 - beta ** self.t)) if self.bias_correction else out

    def _set_val(self, tensor, value):
        tensor.data = Tensor(value, device=tensor.device, dtype=tensor.dtype)
