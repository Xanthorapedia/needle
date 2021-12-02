"""NDDArray backed computation backend.

This backend uses cuda backend_ndarray for cached data and computation.
"""
from needle import backend_ndarray as nd
from needle.device import Device, DLDeviceType
from needle.ops import register_op_attr
import needle.device
import numpy as np


class NDDevice(Device):
    def array(self, array, dtype):
        return nd.array(array, dtype=dtype, device=self.nd_device)

    def empty(self, shape, dtype):
        return nd.empty(shape, dtype=dtype, device=self.nd_device)

    def to_numpy(self, data):
        return data.numpy()

    def fill(self, array, fill_value):
        array.fill(fill_value)
        return array

    def randn(self, shape, dtype, mean=0.0, std=1.0):
        return nd.array(np.random.normal(loc=mean, scale=std, size=shape).astype(dtype), device=self.nd_device)

    def randb(self, shape, dtype, ntrials=1, p=0.5):
        return nd.array(np.random.binomial(ntrials, p, size=shape).astype(dtype), device=self.nd_device)

    def randu(self, shape, dtype, low=0, high=0):
        return nd.array(np.random.uniform(low=low, high=high, size=shape).astype(dtype), device=self.nd_device)

    def one_hot(self, y, num_classes=10):
        #TODO fix this
        y_one_hot = []
        for i in range(y.shape[0]):
            y_one_hot.append(np.eye(num_classes)[int(y[i])])
        y_one_hot = np.array(y_one_hot)
        return nd.array(y_one_hot, device=self.nd_device)

    def enabled(self):
        return self.nd_device.enabled()

    def compute(self, op, inputs, attrs):
        """Dispatch device specific computation"""
        # dispatch device specific compute to op.numpy_compute
        # these computation are registered below.
        return op.nd_compute(inputs, attrs)



class CUDADevice(NDDevice):
    def __init__(self, device_id: int = 0):
        assert device_id == 0
        self.nd_device = nd.cuda()
        self.device_id = device_id

    def __repr__(self):
        return "cuda(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CUDA, self.device_id)

    def __str__(self):
        return self.__repr__()


class CPUDevice(NDDevice):
    def __init__(self, device_id: int = 0):
        self.nd_device = nd.cpu()
        self.device_id = device_id

    def __repr__(self):
        return "cpu(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CPU, self.device_id)

    def __str__(self):
        return self.__repr__()



def cuda(device_id: int = 0) -> CUDADevice:
    return CUDADevice(device_id)


def cpu() -> CPUDevice:
    return CPUDevice()

# set default device to be cpu device.
needle.device._DEFAULT_DEVICE = CPUDevice

def register_nd_compute(name, value=None):
    """Register the compute property based on backend_ndarray
    nd computation can be shared across multiple backends.
    """
    return register_op_attr(name, "nd_compute", value)


# device specific computations
@register_nd_compute("EWiseAdd")
def add(inputs, attrs):
    return inputs[0] + inputs[1]


@register_nd_compute("AddScalar")
def add_scalar(inputs, attrs):
    return inputs[0] + attrs["scalar"]


@register_nd_compute("EWiseMul")
def mul(inputs, attrs):
    return inputs[0] * inputs[1]


@register_nd_compute("MulScalar")
def mul(inputs, attrs):
    return inputs[0] * attrs["scalar"]


@register_nd_compute("EWiseDiv")
def divide(inputs, attrs):
    return inputs[0] / inputs[1]


@register_nd_compute("DivScalar")
def divide_scalar(inputs, attrs):
    return inputs[0] / attrs["scalar"]


@register_nd_compute("PowerScalar")
def power_scalar(inputs, attrs):
    return inputs[0] ** attrs["scalar"]


@register_nd_compute("MatMul")
def matmul(inputs, attrs):
    return inputs[0] @ inputs[1]


@register_nd_compute("Summation")
def summation(inputs, attrs):
    """
    Parameters:
    axes - int or tuple of ints or None

    If axes is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single axis.
    If axes is None, sum over all of the axes.

    Returns an array with the same shape, except with the specified axes removed.
    """
    return inputs[0].sum(axes=attrs["axes"], keepdims=attrs["keepdims"])


@register_nd_compute("BroadcastTo")
def broadcast_to(inputs, attrs):
    return inputs[0].broadcast_to(new_shape=attrs["shape"])


@register_nd_compute("Reshape")
def reshape(inputs, attrs):
    return inputs[0].reshape(attrs["shape"])


@register_nd_compute("Negate")
def negate(inputs, attrs):
    return -inputs[0]


@register_nd_compute("Transpose")
def transpose(inputs, attrs):
    """
    Parameters:
    axes - tuple of ints or None

    If axes is a tuple of ints, permute those two axes.
    If axes is None, permutes the last two axes.
    """
    new_axis = list(range(len(inputs[0].shape)))
    a0, a1 = attrs["axes"] if attrs["axes"] is not None else (-1, -2)
    new_axis[a0], new_axis[a1] = new_axis[a1], new_axis[a0]
    return inputs[0].permute(new_axis)


@register_nd_compute("Log")
def log(inputs, attrs):
    return inputs[0].log()


@register_nd_compute("Exp")
def exp(inputs, attrs):
    return inputs[0].exp()


@register_nd_compute("ReLU")
def relu(inputs, attrs):
    return (inputs[0] > 0) * inputs[0]


@register_nd_compute("LogSoftmax")
def logsoftmax(inputs, attrs):
    """
    Computes log softmax along the last dimension of the array.
    """
    x = inputs[0]
    last = len(x.shape) - 1
    normalized = x - x.max(axes=last, keepdims=True).broadcast_to(x.shape)
    return normalized - normalized.exp().sum(axes=last, keepdims=True).log().broadcast_to(x.shape)


@register_nd_compute("Tanh")
def tanh(inputs, attrs):
    return inputs[0].tanh()


@register_nd_compute("GetItem")
def get_item(inputs, attrs):
    """
    Parameters:
    idxs - indices to index array; tuple of ints or slices

    Returns array indexed by idxs i.e. if array A has shape (5, 3, 2),
    then the shape of the A[0, :, :] would be (3, 2).
    """
    return inputs[0].__getitem__(attrs["idxs"])


@register_nd_compute("SetItem")
def set_item(inputs, attrs):
    """
    Parameters:
    idxs - indices to index array; tuple of ints or slices

    Sets array A at idxs with array B and returns the array.
    """
    return inputs[0].__setitem__(attrs["idxs"], inputs[1])


@register_nd_compute("Stack")
def stack(As, attrs):
    """
    Concatenates a sequence of arrays along a new dimension.

    Parameters:
    axis - dimension to concatenate along

    All arrays need to be of the same size.
    """
    in_shape = As[0].shape
    assert all(in_shape == A.shape for A in As), "All arrays should have the same shape."
    out_shape = list(in_shape)
    out_shape.insert(attrs["axis"], len(As))
    out = nd.empty(tuple(out_shape), dtype=As[0].dtype, device=As[0].device)

    # copy in
    indices = [slice(None)] * len(out.shape)
    for i, A in enumerate(As):
        indices[attrs["axis"]] = i
        out[tuple(indices)] = A

    return out


@register_nd_compute("Flip")
def flip(inputs, attrs):
    """
    Flips the input along specified axes.

    Parameters:
    axes - Axes to flip.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_nd_compute("Dilate")
def dilate(inputs, attrs):
    """
    Dilates the input by a dilation factor on specified axes.
    (i.e., inserts 0s between elements)

    Parameters:
    dilation - Dilation amount (number of 0s to insert)
    axes - Axes to dilate by this amount
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


@register_nd_compute("Conv")
def conv(inputs, attrs):
    """
    Multi-channel 2D convolution of two inputs (called input and weight, respectively).
    inputs[0]: "input", NHWC
    inputs[1]: "weight", (kernel_size, kernel_size, c_in, c_out)

    Parameters:
    padding - (int) Pad the HW axes of the input by this amount
    stride - (int) Stride of the convolution
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION
