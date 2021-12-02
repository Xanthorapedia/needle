from .autograd import Tensor
from . import ops
from .ops import *
# from . import _ffi

# intialize numpy backend
from . import numpy_backend
# from . import cuda_backend

# from .cuda_backend import cuda
from .numpy_backend import numpy_device

from . import nn
from . import optim
from . import init
from . import data
