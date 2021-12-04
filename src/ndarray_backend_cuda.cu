#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
template<typename _VT>
struct CudaVec {
  uint32_t size;
  _VT data[MAX_VEC_SIZE];
};

template<typename _VT>
CudaVec<_VT> VecToCuda(const std::vector<_VT>& x) {
  CudaVec<_VT> shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ size_t inline calc_offset(const CudaVec<int32_t> &strides,
                                     const CudaVec<uint32_t> &index,
                                     size_t offset)
{
  uint32_t loc = offset;
  for (size_t i = 0; i < strides.size; i++)
  {
    loc += strides.data[i] * (int32_t) index.data[i];
  }
  return loc;
}

__device__ void inline calc_index(size_t offset, const CudaVec<uint32_t> &shape, CudaVec<uint32_t> *index)
{
  if (index->size == 0) return;

  CudaVec<int32_t> strides;
  strides.size = shape.size;
  strides.data[strides.size - 1] = 1;

  for (size_t i = index->size - 1; i-- > 0;)
  {
    strides.data[i] = strides.data[i + 1] * shape.data[i + 1];
  }

  for (size_t i = 0; i < strides.size; i++)
  {
    index->data[i] = offset / strides.data[i];
    offset -= index->data[i] * strides.data[i];
  }
}

template<typename _VT>
__device__ void printCudaVec(const CudaVec<_VT> &vec, size_t gid)
{
    if (blockIdx.x * blockDim.x + threadIdx.x == gid)
    {
      printf("%ld: (", gid);
      for (size_t i = 0; i < vec.size; i++)
      {
        printf("%d, ", vec.data[i]);
      }
      printf(")\n");
    }
}

template<typename _VT>
void printCudaVec(const CudaVec<_VT> &vec)
{
  for (size_t i = 0; i < vec.size; i++)
  {
    printf("%d, ", vec.data[i]);
  }
  printf(")\n");
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec<uint32_t> shape,
                              CudaVec<int32_t> strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size)
  {
    CudaVec<uint32_t> index;
    index.size = shape.size;

    calc_index(gid, shape, &index);

    out[gid] = a[calc_offset(strides, index, offset)];
  }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec<uint32_t> shape,
                              CudaVec<int32_t> strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size)
  {
    CudaVec<uint32_t> index;
    index.size = shape.size;

    calc_index(gid, shape, &index);

    out[calc_offset(strides, index, offset)] = a[gid];
  }
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec<uint32_t> shape,
                              CudaVec<int32_t> strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size)
  {
    CudaVec<uint32_t> index;
    index.size = shape.size;

    calc_index(gid, shape, &index);

    out[calc_offset(strides, index, offset)] = val;
  }
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


#define _KER_PREFIX(arg_type) _KER_ARG_##arity##_##arg_type

#define _KER_ARG(arity, arg_type) _KER_ARG_##arity##_##arg_type
#define _KER_ARG_UN_NONE
#define _KER_ARG_BI_ARRY const scalar_t* b,
#define _KER_ARG_BI_SCLR scalar_t val,

#define _KER_APPLY(op, arity, call_type, a, b) _KER_APPLY_##arity##_##call_type(op, (a), (b))
#define _KER_APPLY_BI_CALL(op, a, b) (op((a), (b)))
#define _KER_APPLY_BI_OPER(op, a, b) ((a) op (b))
#define _KER_APPLY_UN_CALL(op, a, b) (op(a))

#define _KER_ARG2(arg_type) _KER_ARG2_##arg_type
#define _KER_ARG2_ARRY b[gid]
#define _KER_ARG2_SCLR val
#define _KER_ARG2_NONE

#define _OP_ARG(arity, arg_type) _OP_ARG_##arity##_##arg_type
#define _OP_ARG_UN_NONE
#define _OP_ARG_BI_ARRY const CudaArray& b,
#define _OP_ARG_BI_SCLR scalar_t val,

#define _OP_ARG2(arg_type) _OP_ARG2_##arg_type
#define _OP_ARG2_ARRY b.ptr,
#define _OP_ARG2_SCLR val,
#define _OP_ARG2_NONE

#define DEFINE_KER(name, op, arity, call_type, arg2_type)                                                \
__global__ void name##Kernel(const scalar_t* a, _KER_ARG(arity, arg2_type) scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
  if (gid < size) out[gid] = _KER_APPLY(op, arity, call_type, a[gid], _KER_ARG2(arg2_type));             \
}

#define DEFINE_OP(name, op, arity, call_type, arg2_type)                            \
DEFINE_KER(name, op, arity, call_type, arg2_type)                                  \
void name(const CudaArray& a, _OP_ARG(arity, arg2_type) CudaArray* out) {           \
  CudaDims dim = CudaOneDim(out->size);                                             \
  name##Kernel<<<dim.grid, dim.block>>>(a.ptr, _OP_ARG2(arg2_type) out->ptr, out->size); \
}

DEFINE_OP(EwiseMul,      *, BI, OPER, ARRY)
DEFINE_OP(ScalarMul,     *, BI, OPER, SCLR)
DEFINE_OP(EwiseDiv,      /, BI, OPER, ARRY)
DEFINE_OP(ScalarDiv,     /, BI, OPER, SCLR)
DEFINE_OP(ScalarPower, pow, BI, CALL, SCLR)

DEFINE_OP(EwiseMaximum,  max, BI, CALL, ARRY)
DEFINE_OP(ScalarMaximum, max, BI, CALL, SCLR)

DEFINE_OP(EwiseEq,  ==, BI, OPER, ARRY)
DEFINE_OP(ScalarEq, ==, BI, OPER, SCLR)
DEFINE_OP(EwiseGe,  >=, BI, OPER, ARRY)
DEFINE_OP(ScalarGe, >=, BI, OPER, SCLR)

DEFINE_OP(EwiseLog,  log,  UN, CALL, NONE)
DEFINE_OP(EwiseExp,  exp,  UN, CALL, NONE)
DEFINE_OP(EwiseTanh, tanh, UN, CALL, NONE)

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__device__ void printMat(scalar_t *mat, size_t H, size_t W)
{
  for (size_t i = 0; i < H; i++)
  {
    for (size_t j = 0; j < W; j++)
    {
      printf("%1.3f, ", mat[i * W + j]);
    }
    printf("\n");
  }
}


__global__ void MatmulKernel(const scalar_t *A, const scalar_t *B, scalar_t *Out, uint32_t M, uint32_t N,
            uint32_t P)
{
  __shared__ scalar_t a[TILE][TILE];
  __shared__ scalar_t b[TILE][TILE];
  __shared__ scalar_t o[TILE][TILE];

  uint32_t tx = threadIdx.x;
  uint32_t ty = threadIdx.y;
  uint32_t ox = blockIdx.x * blockDim.x + tx;
  uint32_t oy = blockIdx.y * blockDim.y + ty;

  o[ty][tx] = 0;

  for (size_t i = 0; i < N; i += TILE)
  {
    // cooperative fetching
    uint32_t ax = i + tx;
    uint32_t by = i + ty;
    uint32_t ay = oy;
    uint32_t bx = ox;

    if (ax < N && ay < M) { a[ty][tx] = A[ay * N + ax]; }
    if (bx < P && by < N) { b[ty][tx] = B[by * P + bx]; }

    __syncthreads();

    // matmul
    for (size_t j = 0; j < TILE && i + j < N; j++) { o[ty][tx] += a[ty][j] * b [j][tx]; }

    __syncthreads();
  }

  if (ox < P && oy < M) { Out[oy * P + ox] = o[ty][tx]; }
}


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  dim3 blockSize(TILE, TILE);
  dim3 gridSize((P + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  MatmulKernel<<<gridSize, blockSize>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////


template<typename OpType>
__global__ void opKernel(const scalar_t *a, size_t out_size, size_t reduce_size, scalar_t * out)
{
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  OpType op;

  if (gid < out_size)
  {
    scalar_t result = op.init;
    for (size_t i = gid * reduce_size; i < (gid + 1) * reduce_size; i++)
    {
      result = op(a[i], result);
    }
    
    out[gid] = result;
  }
}


struct max_op
{
  __device__ scalar_t operator()(scalar_t a, scalar_t b) { return max(a, b); }
  const scalar_t init = -INFINITY;
};



void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

  assert(a.size == out->size * reduce_size);

  CudaDims dim = CudaOneDim(out->size);
  opKernel<max_op><<<dim.grid, dim.block>>>(a.ptr, out->size, reduce_size, out->ptr);
}


struct sum_op
{
  __device__ scalar_t operator()(scalar_t a, scalar_t b) { return a + b; }
  const scalar_t init = 0;
};


void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
   * can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */

  assert(a.size == out->size * reduce_size);

  CudaDims dim = CudaOneDim(out->size);
  opKernel<sum_op><<<dim.grid, dim.block>>>(a.ptr, out->size, reduce_size, out->ptr);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
