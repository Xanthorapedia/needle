"""Core data structures."""
from typing import List, Optional, Dict, TypeVar

import needle
import numpy as np
from .device import default_device, Device, CachedData

LAZY_MODE = False
TENSOR_COUNTER = 0


class Op:
    """Operator definition."""

    def gradient(self, out_grad: "Value", node: "Value") -> List["Value"]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: List[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    attrs: object
    # The following fields are cached fields for
    # dynamic computation
    cached_data: CachedData
    cached_device: Device
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.cached_device.compute(
            self.op, [x.realize_cached_data() for x in self.inputs], self.attrs
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        attrs: object = None,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        cached_device: Device = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        # deduce the device of the computation
        if cached_device is None:
            if not inputs:
                raise ValueError(
                    "Requires cached device to be available for tensor with no inputs"
                )
            cached_device = inputs[0].cached_device
            for x in inputs:
                if cached_device != x.cached_device:
                    raise ValueError(
                        "Requires all input devices to be the same to automatically"
                        "deduce device"
                    )
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)

        self.op = op
        self.inputs = inputs
        self.attrs = attrs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.cached_device = cached_device
        self.requires_grad = requires_grad

    @property
    def device(self):
        return self.cached_device

    @classmethod
    def make_const(cls, data, device, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            cached_device=device,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(
        cls, op: Op, inputs: List["Value"], *, attrs=None, cached_device=None
    ):
        value = cls.__new__(cls)
        value._init(op, inputs, attrs=attrs, cached_device=cached_device)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


class Tuple(Value):
    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.Tuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, Tuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data(), self.device)


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self, array, *, device: Optional[Device] = None, dtype=None, requires_grad=True
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = device.array(array.numpy(), dtype=dtype)
        else:
            device = device if device else default_device()
            cached_data = device.array(array, dtype=dtype)

        self._init(
            None,
            [],
            cached_device=device,
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"], *, attrs=None, cached_device=None):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs, attrs=attrs, cached_device=cached_device)
        if not LAZY_MODE:
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, device, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            cached_device=device,
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.device == self.device and value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data(), self.device)

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else needle.ops.ones_like(self)
        compute_gradient_of_variables(self, out_grad)

    def __getitem__(self, idxs):
        return needle.ops.get_item(self, idxs)

    def __setitem__(self, idxs, other):
        return needle.ops.set_item(self, idxs, other)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        return self.device.to_numpy(self.realize_cached_data())

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.add(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            return needle.ops.add_scalar(self, other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.multiply(self, other)
        else:
            return needle.ops.multiply_scalar(self, other)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return needle.ops.power_scalar(self, other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.add(self, needle.ops.negate(other))
        else:
            return needle.ops.add_scalar(self, -other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.divide(self, other)
        else:
            return needle.ops.divide_scalar(self, other)

    def __matmul__(self, other):
        return needle.ops.matmul(self, other)

    def matmul(self, other):
        return needle.ops.matmul(self, other)

    def sum(self, axes=None, keepdims=False):
        return needle.ops.summation(self, axes, keepdims=keepdims)

    def mean(self, axes=None, keepdims=False):
        axes = tuple(list(range(len(self.shape)))) if axes is None else axes
        n_elem = np.prod([self.shape[ax] for ax in axes])
        return needle.ops.summation(self, axes, keepdims=keepdims) / n_elem

    def broadcast_to(self, shape):
        return needle.ops.broadcast_to(self, shape)

    def reshape(self, shape):
        return needle.ops.reshape(self, shape)

    def __neg__(self):
        return needle.ops.negate(self)

    def transpose(self, axes=None):
        return needle.ops.transpose(self, axes)

    def __gt__(self, other):
        return needle.ops.greater(self, other)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return -self + other

    __rmatmul__ = __matmul__


def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order: List[Tensor] = list(reversed(find_topo_sort([output_tensor])))

    for vi in reverse_topo_order:
        if vi.requires_grad:
            vi.grad = sum_node_list(node_to_output_grads_list[vi])
            # only valid for node with inputs
            if len(vi.inputs) > 0:
                for vj, vji_ in zip(vi.inputs, vi.op.gradient(vi.grad, vi)):
                    node_to_output_grads_list.setdefault(vj, []).append(vji_)


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    visited = {}
    return sum_node_list(topo_sort_dfs(node, visited) for node in node_list)


def topo_sort_dfs(node: Value, visited: Dict[Value, bool]):
    """Post-order DFS"""
    if visited.get(node, False):
        return []
    else:
        visited[node] = True
        return sum([topo_sort_dfs(child, visited) for child in node.inputs], []) + [node]

##############################
####### Helper Methods #######
##############################

_T = TypeVar('_T')

def sum_node_list(node_list: List[_T]) -> _T:
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
