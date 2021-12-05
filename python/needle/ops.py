"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy as np
from .autograd import Op, Tensor, Value, Tuple
from .device import default_device

OP_TABLE = {}


def register_op(name: str, op: Op) -> Op:
    """Register an operator to the op table.

    Parameters
    ----------
    name : str
        The name of the op.

    Returns
    -------
    op : Op
        The registered op.
    """
    if name in OP_TABLE:
        raise ValueError("Op %s is already registered")
    OP_TABLE[name] = op
    return op


def register_op_attr(op_name, attr_name, attr_value=None):
    """Register additional attributes to an existing op by name.


    Parameters
    ----------
    op_name : str
        The name of the op

    attr_name : str
        The name of the attribute

    attr_value :
        The attribute value to be set.

    Returns
    -------
    The attr_value if attr_value is not None.
    Otherwise returns a decorator function.


    Note
    ----
    This function can be used to register additional attributes
    to an Op used by a specific backend.
    """

    def _register(value):
        if op_name not in OP_TABLE:
            raise ValueError("Op %s does not exist")
        op = OP_TABLE[op_name]
        setattr(op, attr_name, value)
        return op

    if attr_value is None:
        return _register
    return _register(attr_value)


class MakeTupleOp(Op):
    def __call__(self, *args: List[Value]) -> Tuple:
        return Tuple.make_from_op(self, list(args))

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, Tuple)
        return [out_grad[i] for i in range(len(out_grad))]


make_tuple = register_op("MakeTuple", MakeTupleOp())


class TupleGetItemOp(Op):
    def __call__(self, a: Tuple, index: int, *, fold_const=True) -> Tensor:
        assert isinstance(a, Tuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTupleOp):
            return a.inputs[index]
        return Tensor.make_from_op(self, [a], attrs={"index": index})

    def gradient(self, out_grad, node):
        index = node.attrs["index"]
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return [make_tuple(*in_grad)]


tuple_get_item = register_op("TupleGetItem", TupleGetItemOp())


class FusedAddScalarsOp(Op):
    def __call__(self, a: Tensor, c0: float, c1: float) -> Tuple:
        return Tuple.make_from_op(self, [a], attrs={"c0": c0, "c1": c1})

    def gradient(self, out_grad, node):
        return [out_grad[0] + out_grad[1]]


fused_add_scalars = register_op("FusedAddScalars", FusedAddScalarsOp())


class EWiseAddOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        return [out_grad, out_grad]


add = register_op("EWiseAdd", EWiseAddOp())


class AddScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad]


add_scalar = register_op("AddScalar", AddScalarOp())


class EWiseMulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs)


multiply = register_op("EWiseMul", EWiseMulOp())


class MulScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad * node.attrs["scalar"]]


multiply_scalar = register_op("MulScalar", MulScalarOp())


class PowerScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        exponent = node.attrs["scalar"]
        return [out_grad * exponent * node.inputs[0] ** (exponent - 1)]
        
power_scalar = register_op("PowerScalar", PowerScalarOp())


class EWiseDivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, -a / (b * b) * out_grad


divide = register_op("EWiseDiv", EWiseDivOp())


class GreaterScalarOp(Op):
    """Op to compare two nodes."""

    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        raise NotImplementedError()


greater = register_op("GreaterScalar", GreaterScalarOp())


class DivScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad / node.attrs["scalar"]]


divide_scalar = register_op("DivScalar", DivScalarOp())


class MatMulOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        a, b = node.inputs
        raw_grad_a, raw_grad_b = out_grad @ b.transpose((-1, -2)), a.transpose((-1, -2)) @ out_grad
        grad_a = raw_grad_a.sum(tuple(range(len(raw_grad_a.shape) - len(a.shape))))
        grad_b = raw_grad_b.sum(tuple(range(len(raw_grad_b.shape) - len(b.shape))))
        return grad_a, grad_b


matmul = register_op("MatMul", MatMulOp())


class SummationOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None, keepdims: bool = False) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes, "keepdims": keepdims})

    def gradient(self, out_grad, node):
        keep_dim_shape = list(node.inputs[0].shape)
        reduced_axes = list(range(len(node.inputs[0].shape))) \
            if node.attrs["axes"] is None else np.atleast_1d(node.attrs["axes"])
        keep_dim_shape = [1 if i in reduced_axes else s for i, s in enumerate(node.inputs[0].shape)]
        return (out_grad.reshape(tuple(keep_dim_shape)).broadcast_to(node.inputs[0].shape),)


summation = register_op("Summation", SummationOp())


class BroadcastToOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        in_shape, out_shape = node.inputs[0].shape, out_grad.shape
        unsqueezed_shape = (1,) * (len(out_shape) - len(in_shape)) + in_shape
        sum_axes = [i for i, (m, n) in enumerate(zip(unsqueezed_shape, out_shape)) if m != n]
        return (out_grad.sum(tuple(sum_axes)).reshape(in_shape),)


broadcast_to = register_op("BroadcastTo", BroadcastToOp())


class ReshapeOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        return (out_grad.reshape(node.inputs[0].shape),)


reshape = register_op("Reshape", ReshapeOp())


class NegateOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return (-out_grad,)


negate = register_op("Negate", NegateOp())


class TransposeOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        return (out_grad.transpose(node.attrs["axes"]),)


transpose = register_op("Transpose", TransposeOp())


class LogOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return (out_grad / node.inputs[0],)


log = register_op("Log", LogOp())


class ExpOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [exp(node.inputs[0]) * out_grad]


exp = register_op("Exp", ExpOp())


class ReLUOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return (Tensor(node.inputs[0].numpy() > 0, device=out_grad.device) * out_grad,)

relu = register_op("ReLU", ReLUOp())


class LogSoftmaxOp(Op):
    def __call__(self, x: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [x])

    def gradient(self, out_grad, node):
        sum_last = summation(out_grad, axes=(-1,), keepdims=True).broadcast_to(out_grad.shape)
        return (out_grad - sum_last * exp(node),)


logsoftmax = register_op("LogSoftmax", LogSoftmaxOp())


class TanhOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return ((1 - tanh(node.inputs[0]) ** 2) * out_grad,)


tanh = register_op("Tanh", TanhOp())


class GetItemOp(Op):
    def __call__(self, a: Tensor, idxs: Tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"idxs": idxs})

    def gradient(self, out_grad, node):
        out = zeros_like(node.inputs[0], device=node.inputs[0].device)
        out = set_item(out, node.attrs["idxs"], out_grad)
        return (out,)

get_item = register_op("GetItem", GetItemOp())


class SetItemOp(Op):
    def __call__(self, a: Tensor, idxs: Tuple, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={"idxs": idxs})

    def gradient(self, out_grad, node):
        raise NotImplementedError()

set_item = register_op("SetItem", SetItemOp())


class StackOp(Op):
    def __call__(self, args: List[Value], axis: int) -> Tensor:
        return Tensor.make_from_op(self, args, attrs={'axis': axis})

    def gradient(self, out_grad, node):
        axis = node.attrs["axis"]
        indices = [slice(None)] * len(out_grad.shape)
        ret = []
        for i in range(out_grad.shape[axis]):
            indices[axis] = i
            ret.append(out_grad.__getitem__(tuple(indices)))
        return tuple(ret)

stack = register_op("Stack", StackOp())


class ConvOp(Op):
    def __call__(self, a: Tensor, b: Tensor, stride: Optional[int] = 1, padding: Optional[int] = 0) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={'stride': stride, 'padding': padding})

    def gradient(self, out_grad, node):
        padding, stride = node.attrs["padding"], node.attrs["stride"]
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)
        (Ph, Pw), (Sh, Sw) = padding, stride

        x, w = node.inputs
        (Kh, Kw) = w.shape[0:2]
        # dispatch gradient into right places
        dilated_out = dilate(out_grad, (Sh - 1, Sw - 1), axes=(1, 2))
        x_grad_padding = Kh - 1 - min(Ph, Kh - 1), Kw - 1 - min(Pw, Kw - 1)
        w_grad_padding = min(Ph, Kh - 1), min(Pw, Kw - 1)
        x_grad = conv(dilated_out, flip(transpose(w, (2, 3)), axes=(0, 1)), padding=x_grad_padding)
        w_grad = conv(transpose(x, (0, 3)), dilate(flip(permute(out_grad, (1, 2, 0, 3)), axes=(0, 1)), (Sh - 1, Sw - 1), axes=(0, 1)), padding=w_grad_padding)

        w_grad = permute(w_grad, (1, 2, 0, 3))
        return (x_grad, w_grad)

conv = register_op("Conv", ConvOp())


class PermuteOp(Op):
    def __call__(self, a: Tensor, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'axes': axes})

    def gradient(self, out_grad, node):
        return (permute(out_grad, tuple(np.argsort(node.attrs["axes"]))),)

permute = register_op("Permute", PermuteOp())


class FlipOp(Op):
    def __call__(self, a: Tensor, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'axes': axes})

    def gradient(self, out_grad, node):
        return (flip(out_grad, node.attrs["axes"]),)

flip = register_op("Flip", FlipOp())


class DilateOp(Op):
    def __call__(self, a: Tensor, dilation: int, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'dilation': dilation, 'axes': axes})

    def gradient(self, out_grad, node):
        dilation, axes, ndim = node.attrs["dilation"], node.attrs["axes"], len(out_grad.shape)
        if isinstance(dilation, int):
            dilation = (dilation,) * ndim
        fill_stride = tuple([d + 1 if i in axes else 1 for i, d in enumerate(dilation)])
        return (out_grad[tuple([slice(None, None, st) for st in fill_stride])],)

dilate = register_op("Dilate", DilateOp())


# additional helper functions
def full(
    shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False
):
    device = device if device else default_device()

    if not rand or "dist" not in rand:
        arr = device.empty(shape, dtype)
        device.fill(arr, fill_value)
    else:
        if rand["dist"] == "normal":
            arr = device.randn(shape, dtype, mean=rand["mean"], std=rand["std"])
        if rand["dist"] == "binomial":
            arr = device.randb(shape, dtype, ntrials=rand["trials"], p=rand["prob"])
        if rand["dist"] == "uniform":
            arr = device.randu(shape, dtype, low=rand["low"], high=rand["high"])

    return Tensor.make_const(arr, device, requires_grad=requires_grad)


def one_hot(labels: Tensor, *, num_classes=10, dtype="float32", device=None):
    device = device if device else default_device()
    arr = device.one_hot(labels.numpy(), num_classes=num_classes)
    return Tensor.make_const(arr, device, requires_grad=False)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def ones(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 1, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, rand={'dist': 'normal', 'mean': mean, 'std': std}, dtype=dtype, device=device, requires_grad=requires_grad)


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, rand={'dist': 'binomial', 'trials': n, 'prob': p}, dtype=dtype, device=device, requires_grad=requires_grad)

def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, rand={'dist': 'uniform', 'low': low, 'high': high}, dtype=dtype, device=device, requires_grad=requires_grad)

def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
