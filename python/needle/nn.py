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
    """A special kind of tensor that represents parameters."""


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
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
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


class Flatten(Module):
    """
    Flattens the dimensions of a Tensor after the first into one dimension.

    Input shape: (bs, s_1, ..., s_n)
    Output shape: (bs, s_1*...*s_n)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


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
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias.reshape((1, self.out_features)).broadcast_to(out.shape)
        return out


class ReLU(Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ndl.relu(x)


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.modules:
            x = mod(x)
        return x


class SoftmaxLoss(Module):
    def __init__(self):
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
        self.weight = Parameter(ndl.ones((self.dim,), dtype=dtype, device=device,
                                         requires_grad=True))
        self.bias = Parameter(ndl.zeros((self.dim,), dtype=dtype, device=device,
                                        requires_grad=True))
        self.running_mean = None
        self.running_var = None

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose((1, -1))

        stats_dims = tuple(list(range(len(x.shape) - 1)))
        N = np.prod([x.shape[d] for d in stats_dims])

        data_shape = (1,) * (len(x.shape) - 1) + (self.dim,)
        x_mean = x.sum(axes=stats_dims) / N
        x_var = ((x - x_mean.reshape(data_shape).broadcast_to(x.shape)) ** 2).sum(axes=stats_dims) / N

        if self.running_mean is None:
            self.running_mean = ndl.zeros_like(x_mean, device=x_mean.device)
            self.running_var = ndl.ones_like(x_var, device=x_var.device)

        if self.training:
            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * x_mean).detach()
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * x_var * (N / (N - 1))).detach()
            mu, sig = x_mean, x_var
        else:
            mu, sig = self.running_mean, self.running_var

        weighted_normalized = (x - mu.reshape(data_shape).broadcast_to(x.shape)) * self.weight.reshape(data_shape).broadcast_to(x.shape) / ((sig.reshape(data_shape) + self.eps) ** 0.5).broadcast_to(x.shape)
        return (weighted_normalized + self.bias.reshape(data_shape).broadcast_to(x.shape)).transpose((1, -1))


class LayerNorm(Module):
    def __init__(self, dims, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        self.weight = Parameter(ndl.ones(self.dims, dtype=dtype, device=device,
                                         requires_grad=True))
        self.bias = Parameter(ndl.zeros(self.dims, dtype=dtype, device=device,
                                        requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        feature_dims = tuple(list(range(len(x.shape) - len(self.dims), len(x.shape))))
        x_demean = x - x.mean(axes=feature_dims, keepdims=True).broadcast_to(x.shape)
        x_var = (x_demean ** 2).mean(axes=feature_dims, keepdims=True).broadcast_to(x.shape)
        data_shape = (1,) * (len(x.shape) - len(self.dims)) + self.dims
        w, b = self.weight.reshape(data_shape).broadcast_to(x.shape), self.bias.reshape(data_shape).broadcast_to(x.shape)
        return x_demean / ((x_var + self.eps) ** 0.5) * w + b


class Dropout(Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.p = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x * ndl.randb(x.shape, n=1, p=(1 - self.p), device=x.device) / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class Flatten(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format

    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same", bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if padding == "same":
            self.padding = (kernel_size - 1) // 2

        self.weight = Parameter(ndl.zeros((kernel_size, kernel_size, in_channels, out_channels)),
                                device=device, dtype=dtype, requires_grad=True)
        init.kaiming_uniform(self.weight)
        if bias:
            self.bias = Parameter(ndl.zeros((out_channels,)),
                                  device=device, dtype=dtype, requires_grad=True)
            k = 1 / (in_channels * kernel_size ** 2) ** 0.5
            init.uniform(self.bias, -k, k)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        x = ndl.permute(x, (0, 2, 3, 1))
        x = ndl.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            x = x + self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(x.shape)
        return ndl.permute(x, (0, 3, 1, 2))


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
