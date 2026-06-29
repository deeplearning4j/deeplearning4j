/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//
#include <array/NDArrayFactory.h>
#include <cusolverDn.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/top_k.h>

#include "execution/Threads.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

// ------------------------------------------------------------------------------------------------------------------ //
//  invert the second diagonal for lower diagonal matrix
template <typename T>
static SD_KERNEL SD_INLINE void invertKernelLow(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
                                      const LongType *inputShape, LongType n) {
  auto inverted = reinterpret_cast<T *>(invertedBuf);
  auto input = reinterpret_cast<const T *>(inputBuf);

  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  for (int i = start + 1; i < n; i += step) {
    LongType pos[] = {i, i - 1};
    LongType posX[] = {i, i};
    LongType posY[] = {i - 1, i - 1};

    LongType xIndex;
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), pos, xIndex);

    LongType dxIndex;
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), posX, dxIndex);

    LongType dyIndex;
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), posY, dyIndex);

    LongType zIndex;
    COORDS2INDEX(shape::rank(invertedShape), shape::stride(invertedShape), pos, zIndex);

    // invert lower triangular matrix
    inverted[zIndex] = -input[xIndex] / (input[dxIndex] * input[dyIndex]);
  }
}
// ------------------------------------------------------------------------------------------------------------------ //
// invert diagonal vals to upper diagonal matrix
template <typename T>
static SD_KERNEL SD_INLINE void upvertKernel(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
                                   const LongType *inputShape, LongType n) {
  auto inverted = reinterpret_cast<T *>(invertedBuf);
  auto input = reinterpret_cast<const T *>(inputBuf);

  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  for (int i = start; i < n; i += step) {
    LongType pos[] = {i, i};
    LongType xIndex, zIndex;
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), pos, xIndex);
    COORDS2INDEX(shape::rank(invertedShape), shape::stride(invertedShape), pos, zIndex);

    // invert diagonal elements
    inverted[zIndex] /= input[xIndex];
  }
}
// ------------------------------------------------------------------------------------------------------------------ //
//  invert upper second diagonal
template <typename T>
static SD_KERNEL SD_INLINE void upvertKernelUp(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
                                     const LongType *inputShape, LongType n) {
  __shared__ T *inverted;
  __shared__ const T *input;
  if (threadIdx.x == 0) {
    inverted = reinterpret_cast<T *>(invertedBuf);
    input = reinterpret_cast<const T *>(inputBuf);
  }
  __syncthreads();

  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  for (int i = start; i < n - 1; i += step) {
    LongType pos[] = {i, i + 1};
    LongType posX[] = {i + 1, i + 1};

    LongType xIndex;
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), pos, xIndex);

    LongType iIndex;
    COORDS2INDEX(shape::rank(invertedShape), shape::stride(invertedShape), posX, iIndex);

    LongType zIndex;
    COORDS2INDEX(shape::rank(invertedShape), shape::stride(invertedShape), pos, zIndex);

    // invert upper matrix
    math::atomics::sd_atomicAdd(&inverted[zIndex], -input[xIndex] * inverted[iIndex]);
  }
}
// ------------------------------------------------------------------------------------------------------------------ //
template <typename T>
static SD_KERNEL SD_INLINE void invertLowKernel(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
                                      const LongType *inputShape, LongType n) {
  auto input = reinterpret_cast<const T *>(inputBuf);
  auto inverted = reinterpret_cast<T *>(invertedBuf);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;

  for (int i = tid + 2; i < n; i += step) {
    for (int j = i - 2; j >= 0; --j)
      for (int k = 0; k < i; k++) {
        LongType posZ[] = {i, j};
        LongType posY[] = {k, j};
        LongType posX[] = {i, k};
        LongType posD[] = {i, i};

        LongType xIndex, yIndex, dIndex, zIndex;
        COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), posX, xIndex);
        COORDS2INDEX(shape::rank(invertedShape), shape::stride(invertedShape), posY, yIndex);
        COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), posD, dIndex);
        COORDS2INDEX(shape::rank(invertedShape), shape::stride(invertedShape), posZ, zIndex);

        // invert non-diagonal elements
        math::atomics::sd_atomicAdd(&inverted[zIndex], -inverted[yIndex] * input[xIndex] / input[dIndex]);
      }
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// Invertion of upper triangular matrix non-diagonal elements when main and second diagonals already processed
template <typename T>
static SD_KERNEL SD_INLINE void invertUpKernel(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
                                     const LongType *inputShape, LongType n) {
  auto inverted = reinterpret_cast<T *>(invertedBuf);
  auto input = reinterpret_cast<const T *>(inputBuf);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int i = (int)n - tid - 2; i >= 0; i -= step) {
    for (int j = i + 2; j < (int)n; j++)
      for (int k = i; k < (int)n; k++) {
        LongType posZ[] = {i, j};
        LongType posY[] = {k, j};
        LongType posX[] = {i, k};

        LongType xIndex, yIndex, zIndex;
        COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), posX, xIndex);
        COORDS2INDEX(shape::rank(invertedShape), shape::stride(invertedShape), posY, yIndex);
        COORDS2INDEX(shape::rank(invertedShape), shape::stride(invertedShape), posZ, zIndex);

        // invert upper non-diagonal elements
        math::atomics::sd_atomicAdd(&inverted[zIndex], -inverted[yIndex] * input[xIndex]);
      }
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// procedure to invert lower-triangular matrix.
// In current case lower triangular matrix has main diagonal with general values
//
template <typename T>
static void invertLowerMatrix_(LaunchContext *context, NDArray *inputMatrix, NDArray *invertedMatrix) {
  int n = inputMatrix->rows();
  invertedMatrix->setIdentity();

  if (inputMatrix->isIdentityMatrix()) return;

  auto stream = context->getCudaStream();

  dim3 lupLaunch = lupDims(n);
  dim3 lupLaunchLow = lupDimsLow(n);
  // invert lower matrix
  // invert main diagonal
  upvertKernel<T><<<lupLaunch.y, lupLaunch.x, lupLaunch.z, *stream>>>(
      invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(),
      inputMatrix->specialShapeInfo(), n);
  sd::DebugHelper::checkErrorCode(stream, "upvertKernel failed");

  // invert the second diagonal
  invertKernelLow<T><<<lupLaunch.y, lupLaunch.x, lupLaunch.z, *stream>>>(
      invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(),
      inputMatrix->specialShapeInfo(), n);

  sd::DebugHelper::checkErrorCode(stream, "invertKernelLow failed");

  // invert non-diagonal elements
  invertLowKernel<T><<<lupLaunchLow.y, lupLaunchLow.x, lupLaunchLow.z, *stream>>>(
      invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(),
      inputMatrix->specialShapeInfo(), n);
  sd::DebugHelper::checkErrorCode(stream, "invertLowKernel failed");
}

// ------------------------------------------------------------------------------------------------------------------ //
// caller for invert lower matrix routine
void invertLowerMatrix(LaunchContext *context, NDArray *inputMatrix, NDArray *invertedMatrix) {
  NDArray::prepareSpecialUse({invertedMatrix}, {inputMatrix});
  BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), invertLowerMatrix_, (context, inputMatrix, invertedMatrix),
                        SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({invertedMatrix}, {inputMatrix});
}

// ------------------------------------------------------------------------------------------------------------------ //
// procedure to invert upper-triangular matrix.
// In current case upper triangular matrix has main diagonal with all ones on it.
template <typename T>
static void invertUpperMatrix_(LaunchContext *context, NDArray *inputMatrix, NDArray *invertedMatrix) {
  int n = inputMatrix->rows();
  invertedMatrix->setIdentity();
  auto stream = context->getCudaStream();
  if (inputMatrix->isIdentityMatrix()) {  // the inverse for I is I
    return;
  }

  // invert upper matrix
  // invert the second diagonal
  upvertKernelUp<T><<<1, n, 512, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(),
                                            inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
  sd::DebugHelper::checkErrorCode(stream, "upvertKernelUp failed");

  // invert other elements
  invertUpKernel<T><<<n, n, 512, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(),
                                            inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
  sd::DebugHelper::checkErrorCode(stream, "invertUpKernel failed");
}

// ------------------------------------------------------------------------------------------------------------------ //
//  invertion of upper triangular matrix - runner routine
void invertUpperMatrix(LaunchContext *context, NDArray *inputMatrix, NDArray *invertedMatrix) {
  NDArray::prepareSpecialUse({invertedMatrix}, {inputMatrix});
  BUILD_SINGLE_SELECTOR(invertedMatrix->dataType(), invertUpperMatrix_, (context, inputMatrix, invertedMatrix),
                        SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({invertedMatrix}, {inputMatrix});
}

// ------------------------------------------------------------------------------------------------------------------ //
// determinant kernel - accumulation product of all values on the main diagonal
template <typename T>
static SD_KERNEL SD_INLINE void determinantKernel(T *compound, T *result, LongType len) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  for (auto i = start; i < len; i += step) {
    auto pos = i * len + i;
    // multiply all diagonal elements
    math::atomics::sd_atomicMul(&result[0], compound[pos]);
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// determinant logarithm - accumulation sum of all logarithm values on the main diagonal. All in logarithic values
// should be positive
template <typename T>
static SD_KERNEL SD_INLINE void determinantLogKernel(T *compound, T *result, LongType len) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  for (auto i = start; i < len; i += step) {
    auto pos = i * len + i;
    // sum logs of all diagonal elements
    math::atomics::sd_atomicAdd(result, math::sd_log<T, T>(math::sd_abs<T,T>(compound[pos])));
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// TAD-aware kernel: copy from a TAD slice of an ND tensor into a contiguous [n,n] matrix buffer.
// tensorBuf + tadOffsets[batchIdx] gives the start of the TAD; tadShape gives the 2D TAD strides.
// matrixBuf is contiguous row-major [n,n].
template <typename T>
static SD_KERNEL SD_INLINE void copyTadToMatrix(const T *tensorBuf, const LongType *tadShape, const LongType *tadOffsets,
                                                T *matrixBuf, LongType batchIdx, LongType n) {
  auto tadPtr = tensorBuf + tadOffsets[batchIdx];
  auto tadStride = shape::stride(tadShape);
  auto n2 = n * n;

  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += blockDim.x * gridDim.x) {
    LongType row = i / n;
    LongType col = i % n;
    LongType coords[] = {row, col};
    LongType tadIdx;
    COORDS2INDEX(2, tadStride, coords, tadIdx);
    matrixBuf[i] = tadPtr[tadIdx];
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// TAD-aware kernel: copy from a contiguous [n,n] matrix buffer back into a TAD slice of an ND tensor.
template <typename T>
static SD_KERNEL SD_INLINE void copyMatrixToTad(const T *matrixBuf, T *tensorBuf, const LongType *tadShape,
                                                const LongType *tadOffsets, LongType batchIdx, LongType n) {
  auto tadPtr = tensorBuf + tadOffsets[batchIdx];
  auto tadStride = shape::stride(tadShape);
  auto n2 = n * n;

  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += blockDim.x * gridDim.x) {
    LongType row = i / n;
    LongType col = i % n;
    LongType coords[] = {row, col};
    LongType tadIdx;
    COORDS2INDEX(2, tadStride, coords, tadIdx);
    tadPtr[tadIdx] = matrixBuf[i];
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// Padded copy kernel: copies n×n elements between two contiguous C-order matrices with different row strides.
// srcBatchStride / dstBatchStride = number of elements between consecutive batch matrices.
// srcRowStride / dstRowStride = number of elements between consecutive rows within a matrix.
// Copies only the n×n top-left block (for padding smaller into larger matrices).
template <typename T>
static SD_KERNEL void copyPaddedBatch(const T *srcBuf, LongType srcBatchStride, LongType srcRowStride,
                                      T *dstBuf, LongType dstBatchStride, LongType dstRowStride,
                                      LongType batchIdx, LongType n) {
  auto srcPtr = srcBuf + batchIdx * srcBatchStride;
  auto dstPtr = dstBuf + batchIdx * dstBatchStride;
  auto n2 = n * n;

  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += blockDim.x * gridDim.x) {
    LongType row = i / n;
    LongType col = i % n;
    dstPtr[row * dstRowStride + col] = srcPtr[row * srcRowStride + col];
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// fill up permutaion matrix kernel. Permutation matrix filled with zeros and ones
template <typename F>
static SD_KERNEL SD_INLINE void fillUpPermutation(void *output, const LongType *shape, int *source, int rowNum) {
  F *permutation = reinterpret_cast<F *>(output);

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  for (auto i = start; i < rowNum; i += step) {
    int val = source[i] - 1;
    LongType posF[] = {i, val};
    LongType pos;
    COORDS2INDEX(shape::rank(shape), shape::stride(shape), posF, pos);
    permutation[pos] = F(1.f);
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// LUP decomposition runner - using CUBLAS SOLVER
// if permutation is given, then using LUP decomposition, LU decomposition otherwise
// L - lower triangular, U - upper triangular, P - permutation matrices
// PA = LU
//
// input - A matrix nxn
// compound - C matrix L + U - I, or main diagonal and lower - L matrix, from the 2nd diagonal - U matrix
template <typename T, typename I>
static void lup_(LaunchContext *context, NDArray *input, NDArray *compound, NDArray *permutation) {
  auto stream = context->getCudaStream();
  auto n = input->rows();
  std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

  cusolverDnHandle_t *cusolverH = (cusolverDnHandle_t *)context->getCusolverHandle();  // nullptr;
  // create solver handle
  cusolverStatus_t status;

  // set solver stream
  status = cusolverDnSetStream(*cusolverH, *stream);
  if (CUSOLVER_STATUS_SUCCESS != status) {
    { std::string msg = "Cannot set up stream for cuda solver; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }
  int lwork = 0;
  int *d_info = nullptr;
  // allocate memory for permutation vector
  int lupDevId = 0; cudaGetDevice(&lupDevId);
  d_info = reinterpret_cast<int*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(LongType), lupDevId, nullptr));
  if (d_info == nullptr) THROW_EXCEPTION("helpers::lup_: Cannot allocate memory for solver info buffer");
  cudaError_t err;

  DataType dtype = input->dataType();
  switch (dtype) {  // there are two implementations with cublas for LUP decomposition - double and float

    case DOUBLE: {
      double *d_work = nullptr;
      // compute internal buffer size
      double *matrix = reinterpret_cast<double *>(input->specialBuffer());
      status = cusolverDnDgetrf_bufferSize(*cusolverH, n, n, matrix, n, &lwork);
      if (CUSOLVER_STATUS_SUCCESS != status) {
        { std::string msg = "helpers::lup_: Cannot create cuSolver handle; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
      }

      d_work = reinterpret_cast<double*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(double) * lwork, lupDevId, nullptr));
      if (d_work == nullptr) THROW_EXCEPTION("helpers::lup_: Cannot allocate memory for solver data buffer");

      if (permutation == nullptr) {
        status = cusolverDnDgetrf(*cusolverH, n, n, matrix, n, d_work, nullptr, d_info);

        if (status != CUSOLVER_STATUS_SUCCESS) {
          { std::string msg = "helpers::lup_: LU factorization is failed due ; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
        }
      } else {
        std::vector<LongType> shape = {n};
        NDArray permutVector('c', shape, INT32, context);
        int *permutationBuf = permutVector.dataBuffer()->specialAsT<int>();
        status = cusolverDnDgetrf(*cusolverH, n, n, matrix, n, d_work, permutationBuf, d_info);
        if (status != CUSOLVER_STATUS_SUCCESS) {
          { std::string msg = "helpers::lup_: LU factorization is failed due ; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
        }

        if (permutation->rankOf() == 2) {
          fillUpPermutation<double><<<n, n, 1024, *stream>>>(permutation->specialBuffer(),
                                                             permutation->specialShapeInfo(), permutationBuf, n);
          sd::DebugHelper::checkErrorCode(stream, "fillUpPermutation failed");

        } else {
          // cuSolver wrote permutVector and input on device; register them so assign's
          // internal D2H sync sees the correct device data before copying.
          NDArray::registerSpecialUse({&permutVector, input}, {});
          compound->assign(input);
          permutation->assign(&permutVector);
        }
      }
      sd::memory::CudaMemoryPool::getInstance().free(d_work, lupDevId, nullptr);
    } break;
    case FLOAT32: {
      float *matrix = reinterpret_cast<float *>(input->specialBuffer());
      float *d_work = nullptr;

      status = cusolverDnSgetrf_bufferSize(*cusolverH, n, n, matrix, n, &lwork);
      if (CUSOLVER_STATUS_SUCCESS != status) {
        { std::string msg = "helpers::lup_: Cannot create cuSolver handle; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
      }

      d_work = reinterpret_cast<float*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(float) * lwork, lupDevId, nullptr));
      if (d_work == nullptr) THROW_EXCEPTION("helpers::lup_: Cannot allocate memory for solver data buffer (float)");

      if (permutation == nullptr)
        status = cusolverDnSgetrf(*cusolverH, n, n, matrix, n, d_work, nullptr, d_info);
      else {
        std::vector<LongType> shape = {n};
        NDArray permutVector('c', shape, INT32, context);
        int *permutationBuf = reinterpret_cast<int *>(permutVector.specialBuffer());
        status = cusolverDnSgetrf(*cusolverH, n, n, matrix, n, d_work, permutationBuf, d_info);
        if (permutation->rankOf() == 2) {
          fillUpPermutation<I><<<n, n, 128, *stream>>>(permutation->specialBuffer(), permutation->specialShapeInfo(),
                                                       permutationBuf, n);
          sd::DebugHelper::checkErrorCode(stream, "fillUpPermutation failed");

          // fillUpPermutation kernel wrote permutation on device; register it now.
          NDArray::registerSpecialUse({permutation}, {});
        } else {
          // cuSolver wrote permutVector and input on device; register them so assign's
          // internal D2H sync sees the correct device data before copying.
          NDArray::registerSpecialUse({&permutVector, input}, {});
          compound->assign(input);
          permutation->assign(&permutVector);
        }
      }
      sd::memory::CudaMemoryPool::getInstance().free(d_work, lupDevId, nullptr);
    }
  }
  if (CUSOLVER_STATUS_SUCCESS != status) {
    { std::string msg = "helpers::lup_: Cannot make LU decomposition; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }
  sd::memory::CudaMemoryPool::getInstance().free(d_info, lupDevId, nullptr);

  // cuSolver getrf wrote input in-place on device; register it unconditionally.
  NDArray::registerSpecialUse({input}, {});
}
// ------------------------------------------------------------------------------------------------------------------ //

BUILD_DOUBLE_TEMPLATE( void lup_,
                      (LaunchContext * context, NDArray *input, NDArray *output, NDArray *permutation), SD_FLOAT_NATIVE,
                      SD_INDEXING_TYPES);

template <typename T>
static void swapRows_(NDArray *matrix, LongType theFirst, LongType theSecond) {
  if (theFirst != theSecond)
    for (LongType i = 0; i < matrix->columns(); i++) {
      math::sd_swap(matrix->r<T>(theFirst, i), matrix->r<T>(theSecond, i));
    }
}
BUILD_SINGLE_TEMPLATE( void swapRows_, (NDArray * matrix, sd::LongType theFirst, sd::LongType theSecond),
                      SD_FLOAT_TYPES);

void swapRows(NDArray *matrix, LongType theFirst, LongType theSecond) {
  BUILD_SINGLE_SELECTOR(matrix->dataType(), swapRows_, (matrix, theFirst, theSecond), SD_FLOAT_TYPES);
}

template <typename T>
void processColumns(LongType currentRow, LongType rowNum, T *compoundBuf, LongType const *compoundShape) {
  LongType xDiag[] = {currentRow, currentRow};
  LongType diagIndex;
  COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), xDiag, diagIndex);

  auto loop = PRAGMA_THREADS_FOR {
    for (auto j = start; j < stop; j++) {
      LongType xRow[] = {j, currentRow};
      LongType rowIndex;
      COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), xRow, rowIndex);
      compoundBuf[rowIndex] /= compoundBuf[diagIndex];  // output->t<T>(i, i);

      for (LongType k = currentRow + 1; k < rowNum; k++) {
        LongType yRow[] = {j, k};
        LongType yCol[] = {currentRow, k};
        LongType rowIndexY, colIndex;
        COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), yRow, rowIndexY);
        COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), yCol, colIndex);
        compoundBuf[rowIndexY] -= compoundBuf[rowIndex] * compoundBuf[colIndex];
      }
    }
  };
  samediff::Threads::parallel_tad(loop, currentRow + 1, rowNum, 1);
}
// #define INSTANTIATE_PROCESS_COLUMNS(T) template void processColumns<GET_SECOND(T)>(LongType currentRow, LongType rowNum, GET_SECOND(T) *compoundBuf, LongType const *compoundShape);
// ITERATE_LIST((SD_FLOAT_NATIVE), INSTANTIATE_PROCESS_COLUMNS)
template void processColumns<float>(LongType currentRow, LongType rowNum, float *compoundBuf, LongType const *compoundShape);
template void processColumns<double>(LongType currentRow, LongType rowNum, double *compoundBuf, LongType const *compoundShape);
template void processColumns<float16>(LongType currentRow, LongType rowNum, float16 *compoundBuf, LongType const *compoundShape);

template <typename T>
static void swapRows(T *matrixBuf, LongType const *matrixShape, LongType theFirst, LongType theSecond) {
  if (theFirst != theSecond) {
    auto n = shape::sizeAt(matrixShape, static_cast<LongType>(-1));

    auto loop = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        LongType theFirstPos[] = {theFirst, i};
        LongType theSecondPos[] = {theSecond, i};
        LongType theFirstIndex, theSecondIndex;
        COORDS2INDEX(shape::rank(matrixShape), shape::stride(matrixShape), theFirstPos, theFirstIndex);
        COORDS2INDEX(shape::rank(matrixShape), shape::stride(matrixShape), theSecondPos, theSecondIndex);
        math::sd_swap(matrixBuf[theFirstIndex], matrixBuf[theSecondIndex]);
      }
    };

    samediff::Threads::parallel_tad(loop, 0, n, 1);
  }
}
template <typename T>
static void doolitleLU(LaunchContext *context, NDArray *compound, LongType rowNum) {
  auto input = compound->dup();
  compound->nullify();

  // Decomposing matrix into Upper and Lower
  // triangular matrix
  for (auto i = 0; i < rowNum; i++) {
    // Upper Triangular
    for (auto k = i; k < rowNum; k++) {
      // Summation of L(i, j) * U(j, k)
      LongType sum = 0;
      for (LongType j = 0; j < i; j++) sum += compound->t<T>(i, j) * compound->t<T>(j, k);

      // Evaluating U(i, k)
      compound->r<T>(i, k) = input->t<T>(i, k) - sum;
    }

    // Lower Triangular
    for (LongType k = i + 1; k < rowNum; k++) {
      // Summation of L(k, j) * U(j, i)
      LongType sum = 0;
      for (LongType j = 0; j < i; j++) sum += compound->t<T>(k, j) * compound->t<T>(j, i);

      // Evaluating L(k, i)
      compound->r<T>(k, i) = (input->t<T>(k, i) - sum) / compound->t<T>(i, i);
    }
  }

  delete input;
}

/*
 * lu decomposition with naive algorithm with partial pivoting
 * */
template <typename T, typename I>
static I argmaxCol(I column, T* compoundBuffer, sd::LongType const* compoundShape) {
  auto rowNum = shape::sizeAt(compoundShape, static_cast<sd::LongType>(0));
  sd::LongType xInitial[] = {column, column};
  sd::LongType xInitialIndex;
  COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), xInitial, xInitialIndex);
  auto maxValue = T(0);
  auto result = -1;
  auto start = column;
  auto stop = rowNum;
  auto increment = 1;
  for (auto rowCounter = start; rowCounter < stop; rowCounter++) {
    sd::LongType xPos[] = {rowCounter, column};
    sd::LongType xIndex;
    COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), xPos, xIndex);
    if (sd::math::sd_abs<T,T>(compoundBuffer[xIndex]) > maxValue) {
      T absVal = sd::math::sd_abs<T,T>(compoundBuffer[xIndex]);
      maxValue = maxValue > absVal ? maxValue : absVal;
      result = rowCounter;
    }
  }

  return result;
}

template <typename T, typename I>
static void luNN_(LaunchContext *context, NDArray *compound, NDArray *permutation, LongType rowNum) {
  // compound is read+written (Gaussian elimination reads existing values, then overwrites);
  // permutation is written (linspace + pivot swaps). BOTH go in the write list so the standard
  // registerPrimaryUse call below ticks their host writes — coherence is handled by prepare/
  // register ONLY, never manual ticks. compound is also in the read list so it syncs device→host
  // before bufferAsT<T>() reads it.
  NDArray::preparePrimaryUse({compound, permutation}, {compound});

  if (permutation) {  // LUP algorithm
    permutation->linspace(0);

    // Cache rank, shape, and stride values
    sd::LongType permRank = shape::rank(permutation->shapeInfo());
    const sd::LongType* permShape = shape::shapeOf(permutation->shapeInfo());
    const sd::LongType* permStride = shape::stride(permutation->shapeInfo());

    auto permutationBuf = permutation->bufferAsT<I>();
    auto compoundBuf = compound->bufferAsT<T>();
    auto compoundShape = compound->shapeInfo();

    for (LongType i = 0; i < rowNum - 1; i++) {
      auto pivotIndex = argmaxCol(i, compoundBuf, compoundShape);
      if (pivotIndex < 0) {
        THROW_EXCEPTION("helpers::luNN_: input matrix is singular.");
      }

      // Precompute coordinates and offsets for permutation swaps
      sd::LongType permIndex1, permIndex2;
      sd::LongType permCoords1[SD_MAX_RANK], permCoords2[SD_MAX_RANK];

      INDEX2COORDS(i, permRank, permShape, permCoords1);
      COORDS2INDEX(permRank, permStride, permCoords1, permIndex1);

      INDEX2COORDS(pivotIndex, permRank, permShape, permCoords2);
      COORDS2INDEX(permRank, permStride, permCoords2, permIndex2);

      // Swap permutation elements
      math::sd_swap(permutationBuf[permIndex1], permutationBuf[permIndex2]);

      // Swap rows in the compound matrix
      swapRows(compoundBuf, compoundShape, i, pivotIndex);

      // Process the columns for LU decomposition
      processColumns(i, rowNum, compoundBuf, compoundShape);
    }
    // permutation's host writes (linspace + pivot swaps) are registered by registerPrimaryUse
    // below — coherence is handled by the standard prepare/register calls, never manual ticks.
  } else {  // Doolittle algorithm with LU decomposition
    doolitleLU<T>(context, compound, rowNum);
  }

  NDArray::registerPrimaryUse({compound, permutation}, {});
}


template <typename T, typename I>
static void lu_(LaunchContext *context, NDArray *input, NDArray *output, NDArray *permutationVectors) {
  NDArray::preparePrimaryUse({output}, {input, permutationVectors});

  auto n = input->sizeAt(-1);

  output->assign(input);  // copy input data to output

  // For unbatched (2D) inputs, allTensorsAlongDimension({-2,-1}) produces rank-0 TADs
  // which breaks coordinate-based indexing in luNN_. Process the single matrix directly.
  if (input->rankOf() == 2) {
    luNN_<T, I>(context, output, permutationVectors, n);
    NDArray::registerPrimaryUse({output}, {input, permutationVectors});
    // Host wrote output; push to device so callers using specialBuffer() see the result.
    NDArray::prepareSpecialUse({output}, {output});
    NDArray::registerSpecialUse({output}, {});
    return;
  }

  ResultSet outputs = output->allTensorsAlongDimension({-2, -1});
  ResultSet permutations;
  if (permutationVectors) permutations = permutationVectors->allTensorsAlongDimension({-1});
  auto loop = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      luNN_<T, I>(context, outputs.at(i), permutationVectors ? permutations.at(i) : nullptr, n);
    }
  };
  samediff::Threads::parallel_for(loop, 0, outputs.size(), 1);
  NDArray::registerPrimaryUse({output}, {input, permutationVectors});
  // Host wrote output; push to device so callers using specialBuffer() see the result.
  NDArray::prepareSpecialUse({output}, {output});
  NDArray::registerSpecialUse({output}, {});
}

void lu(LaunchContext *context, NDArray *input, NDArray *output, NDArray *permutations) {
  BUILD_DOUBLE_SELECTOR(input->dataType(), permutations->dataType(), lu_, (context, input, output, permutations),
                        SD_FLOAT_NATIVE, SD_INDEXING_TYPES);
}
// ------------------------------------------------------------------------------------------------------------------ //
template <typename T>
static Status determinant_(LaunchContext *context, NDArray *input, NDArray *output) {
  LongType n = input->sizeAt(-1);
  LongType n2 = n * n;
  std::vector<LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &dims2);
  const LongType batchSize = packX->numberOfTads();

  auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, DataTypeUtils::fromT<T>(), context);
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input});
  dim3 launchDims = getLaunchDims("logAbsDeterminant");
  float one = 1.f;
  output->assign(one);

  auto inputBuf = reinterpret_cast<const T*>(input->specialBuffer());

  // Cache rank, shape, and stride outside the loop
  sd::LongType outputRank = shape::rank(output->shapeInfo());
  const sd::LongType* outputShape = shape::shapeOf(output->shapeInfo());
  const sd::LongType* outputStride = shape::stride(output->shapeInfo());

  for (LongType e = 0; e < batchSize; e++) {
    copyTadToMatrix<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        inputBuf, packX->specialShapeInfo(), packX->specialOffsets(),
        reinterpret_cast<T*>(matrix->specialBuffer()), e, n);
    sd::DebugHelper::checkErrorCode(stream, "copyTadToMatrix failed");

    lup_<T, int>(context, matrix, nullptr, nullptr);

    // Precompute coordinates and offsets
    LongType offsetCoords[SD_MAX_RANK];
    LongType offset;
    INDEX2COORDS(e, outputRank, outputShape, offsetCoords);
    COORDS2INDEX(outputRank, outputStride, offsetCoords, offset);

    // Initialize output to 1.0 before atomic multiplication
    T initVal = static_cast<T>(1);
    auto outputBuf = reinterpret_cast<T*>(output->specialBuffer()) + offset;
    cudaMemcpyAsync(outputBuf, &initVal, sizeof(T), cudaMemcpyHostToDevice, *stream);
    // During CUDA graph capture, synchronous calls are illegal.
    if (!tl_graphExecutionActive && !tl_dspReplayActive) { cudaStreamSynchronize(*stream); }

    determinantKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        reinterpret_cast<T*>(matrix->specialBuffer()), outputBuf, n);
    sd::DebugHelper::checkErrorCode(stream, "determinantKernel failed");
  }

  delete matrix;
  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}


BUILD_SINGLE_TEMPLATE(Status determinant_, (LaunchContext *context, NDArray *input, NDArray *output), SD_FLOAT_NATIVE);

Status determinant(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), return determinant_, (context, input, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
Status logAbsDeterminant_(LaunchContext *context, NDArray *input, NDArray *output) {
  LongType n = input->sizeAt(-1);
  LongType n2 = n * n;
  std::vector<LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};
  DataType dtype = input->dataType();
  if (dtype != DOUBLE) dtype = FLOAT32;

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &dims2);
  const LongType batchSize = packX->numberOfTads();

  auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, dtype, context);
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input});
  dim3 launchDims = getLaunchDims("logAbsDeterminant");
  float zero = 0.f;
  output->assign(zero);

  auto inputBuf = reinterpret_cast<const T*>(input->specialBuffer());

  // Cache rank, shape, and stride outside the loop
  sd::LongType outputRank = shape::rank(output->shapeInfo());
  const sd::LongType* outputShape = shape::shapeOf(output->shapeInfo());
  const sd::LongType* outputStride = shape::stride(output->shapeInfo());

  for (LongType e = 0; e < batchSize; e++) {
    copyTadToMatrix<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        inputBuf, packX->specialShapeInfo(), packX->specialOffsets(),
        reinterpret_cast<T*>(matrix->specialBuffer()), e, n);
    sd::DebugHelper::checkErrorCode(stream, "copyTadToMatrix failed");

    lup_<T, int>(context, matrix, nullptr, nullptr);

    // Precompute coordinates and offsets
    LongType offsetCoords[SD_MAX_RANK];
    LongType offset;
    INDEX2COORDS(e, outputRank, outputShape, offsetCoords);
    COORDS2INDEX(outputRank, outputStride, offsetCoords, offset);

    auto outputBuf = reinterpret_cast<T *>(output->specialBuffer()) + offset;
    determinantLogKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        reinterpret_cast<T*>(matrix->specialBuffer()), outputBuf, n);
    sd::DebugHelper::checkErrorCode(stream, "determinantLogKernel failed");
  }

  delete matrix;
  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}


BUILD_SINGLE_TEMPLATE(Status logAbsDeterminant_, (LaunchContext *context, NDArray *input, NDArray *output), SD_FLOAT_NATIVE);

Status logAbsDeterminant(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), return logAbsDeterminant_, (context, input, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
static SD_KERNEL SD_INLINE void fillLowerUpperKernel(void *lowerBuf, const LongType *lowerShape, void *upperBuf,
                                           const LongType *upperShape, void *matrixBuf, const LongType *matrixShape,
                                           LongType n) {
  __shared__ T *lowerMatrix;
  __shared__ T *upperMatrix;
  __shared__ T *matrix;

  if (threadIdx.x == 0) {
    lowerMatrix = reinterpret_cast<T *>(lowerBuf);
    upperMatrix = reinterpret_cast<T *>(upperBuf);
    matrix = reinterpret_cast<T *>(matrixBuf);
  }
  __syncthreads();

  for (int k = blockIdx.x; k < n; k += gridDim.x) {  // and then put all values under main diagonal on to it
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      LongType posX[] = {k, j};
      LongType posD[] = {j, j};
      LongType xPos, yPos, iPos, dPos;
      COORDS2INDEX(shape::rank(lowerShape), shape::stride(lowerShape), posX, xPos);
      COORDS2INDEX(shape::rank(upperShape), shape::stride(upperShape), posX, yPos);
      COORDS2INDEX(shape::rank(matrixShape), shape::stride(matrixShape), posX, iPos);
      COORDS2INDEX(shape::rank(matrixShape), shape::stride(matrixShape), posD, dPos);
      if (k >= j)
        lowerMatrix[xPos] = matrix[iPos];  //(k, j);
      else
        upperMatrix[yPos] = matrix[iPos];  // k, j);
    }
  }
}
template <typename T>
static Status inverse_(LaunchContext *context, NDArray *input, NDArray *output) {
  auto n = input->sizeAt(-1);
  auto n2 = n * n;
  auto dtype = DataTypeUtils::fromT<T>();

  auto matrix = NDArrayFactory::create('c', {n, n}, dtype, context);
  auto upper = NDArrayFactory::create('c', {n, n}, dtype, context);
  auto lower = NDArrayFactory::create('c', {n, n}, dtype, context);
  auto compound = NDArrayFactory::create('c', {n, n}, dtype, context);
  auto permutation = NDArrayFactory::create('c', {n, n}, dtype, context);

  std::vector<LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};
  std::vector<LongType> dims3 = {output->rankOf() - 2, output->rankOf() - 1};

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &dims2);
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dims3);

  auto stream = context->getCudaStream();
  auto inputBuf = reinterpret_cast<const T*>(input->specialBuffer());
  auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  dim3 launchDims = getLaunchDims("logAbsDeterminant");

  for (LongType i = 0; i < packX->numberOfTads(); i++) {
    copyTadToMatrix<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        inputBuf, packX->specialShapeInfo(), packX->specialOffsets(),
        reinterpret_cast<T*>(matrix->specialBuffer()), i, n);
    sd::DebugHelper::checkErrorCode(stream, "copyTadToMatrix failed");
    // copyTadToMatrix kernel wrote matrix on device; register before lup_ reads specialBuffer().
    std::vector<NDArray*> matrixOnly = {matrix};
    std::vector<NDArray*> lowerUpper = {lower, upper};
    NDArray::registerSpecialUse(matrixOnly, {});
    lup_<T, int>(context, matrix, nullptr, nullptr);
    // lup_ already registers matrix as device-written; prepare lower+upper as device-read
    // inputs for fillLowerUpperKernel which also writes them.
    NDArray::prepareSpecialUse(lowerUpper, matrixOnly);
    fillLowerUpperKernel<T><<<n, n, 1024, *stream>>>(lower->specialBuffer(), lower->specialShapeInfo(),
                                                     upper->specialBuffer(), upper->specialShapeInfo(),
                                                     matrix->specialBuffer(), matrix->specialShapeInfo(), n);
    sd::DebugHelper::checkErrorCode(stream, "fillLowerUpperKernel failed");
    NDArray::registerSpecialUse(lowerUpper, matrixOnly);

    int zero = 0;
    matrix->assign(zero);
    invertUpperMatrix(context, upper, matrix);  // U^{-1} — wrapper handles prepare/register
    compound->assign(zero);
    invertLowerMatrix(context, lower, compound);  // L^{-1} — wrapper handles prepare/register

    MmulHelper::mmul(matrix, compound, upper, 1.0, 0.0);  // upper = matrix * compound; mmul handles prepare/register
    copyMatrixToTad<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        reinterpret_cast<const T*>(upper->specialBuffer()), outputBuf,
        packZ->specialShapeInfo(), packZ->specialOffsets(), i, n);
    sd::DebugHelper::checkErrorCode(stream, "copyMatrixToTad failed");
  }

  delete matrix;
  delete upper;
  delete lower;
  delete compound;
  delete permutation;
  return Status::OK;
}

Status inverse(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), return inverse_, (context, input, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {input});
}

bool checkCholeskyInput(LaunchContext *context, NDArray *input) { return true; }

template <typename F>
SD_KERNEL SD_INLINE void fillBatchKernel(F **dArrayBatch, F *buf, const LongType *offsets, LongType batchSize) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (auto i = start; i < batchSize; i += step) {
    dArrayBatch[i] = buf + offsets[i];
  }
}

template <typename F>
SD_KERNEL SD_INLINE void adjustResultsKernel(F *dArray, const LongType *shape, const LongType *offsets, LongType batchSize,
                                   LongType n) {
  // auto i = blockIdx.x * blockDim.x + threadIdx.x;
  LongType *shapeOf = shape::shapeOf(shape);
  LongType *strideOf = shape::stride(shape);

  for (auto i = blockIdx.x; i < batchSize; i += gridDim.x) {
    auto current = dArray + offsets[i];
    for (auto r = threadIdx.x; r < n; r += blockDim.x) {
      for (auto c = r + 1; c < n; c++) {
        LongType posRC[] = {r, c};
        auto pos = r * n + c;
        current[pos] = 0.;
      }
    }
  }
}
// Explicit template instantiations for CUDA kernel functions
template SD_KERNEL SD_INLINE void fillBatchKernel<float>(float **dArrayBatch, float *buf, const LongType *offsets, LongType batchSize);
template SD_KERNEL SD_INLINE void fillBatchKernel<double>(double **dArrayBatch, double *buf, const LongType *offsets, LongType batchSize);
template SD_KERNEL SD_INLINE void adjustResultsKernel<float>(float *dArray, const LongType *shape, const LongType *offsets, LongType batchSize, LongType n);
template SD_KERNEL SD_INLINE void adjustResultsKernel<double>(double *dArray, const LongType *shape, const LongType *offsets, LongType batchSize, LongType n);

template <typename F>
Status cholesky__(LaunchContext *context, NDArray *input, NDArray *output, bool inplace) {
  if (!inplace) output->assign(input);
  auto tempOutput = output->dup();
  cusolverDnHandle_t handle = nullptr;
  auto n = input->sizeAt(-1);
  auto n2 = n * n;
  NDArray::prepareSpecialUse({output}, {input});

  auto status = cusolverDnCreate(&handle);
  if (CUSOLVER_STATUS_SUCCESS != status) {
    { std::string msg = "helpers::cholesky_: Cannot create solver handle; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }
  F **dArrayBatch = nullptr;
  // Compute batch size directly: product of all dims except the last two.
  // TAD along {rank-2, rank-1} on rank-2 arrays produces per-element scalar TADs
  // (batchSize = n*n) because dimsToExclude is empty, which is wrong for batch
  // matrix ops that need batchSize = 1 for a single matrix.
  const LongType rank = tempOutput->rankOf();
  LongType batchSize = 1;
  for (LongType d = 0; d < rank - 2; d++) {
    batchSize *= tempOutput->sizeAt(d);
  }
  int *dInfoArray = nullptr;
  int cholDevId = 0; cudaGetDevice(&cholDevId);
  auto stream = context->getCudaStream();
  dArrayBatch = reinterpret_cast<F**>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(F *) * batchSize, cholDevId, *stream));
  if (dArrayBatch == nullptr) THROW_EXCEPTION("helpers::cholesky_: Cannot allocate memory for solver batch data buffer");
  dInfoArray = reinterpret_cast<int*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(LongType) * batchSize, cholDevId, *stream));
  if (dInfoArray == nullptr) THROW_EXCEPTION("helpers::cholesky_: Cannot allocate memory for solver errors buffer");
  // Build batch pointer array on host: each n×n matrix is n2 elements apart.
  {
    auto baseBuf = reinterpret_cast<F*>(tempOutput->specialBuffer());
    std::vector<F*> hostBatchPtrs(batchSize);
    for (LongType i = 0; i < batchSize; i++) {
      hostBatchPtrs[i] = baseBuf + i * n2;
    }
    cudaMemcpyAsync(dArrayBatch, hostBatchPtrs.data(), sizeof(F*) * batchSize,
                    cudaMemcpyHostToDevice, *stream);
  }

  status = cusolverDnSetStream(handle, *stream);
  if (CUSOLVER_STATUS_SUCCESS != status) {
    { std::string msg = "helpers::cholesky_: Cannot set stream to solver handle; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  // cuSOLVER's internal potrfBatched kernel uses 32-element tiles. When n < 32 it reads
  // past the n×n matrix. Pad each matrix to lda=32 so the tiling never OOBs.
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  const int lda = (n < 32) ? 32 : static_cast<int>(n);

  if (n < 32) {
    // cuSOLVER potrfBatched uses 32-element tiles internally — OOB when lda < 32.
    // Create a padded NDArray [batchSize, lda, lda] so TAD gives correct offsets,
    // then use fillBatchKernel + potrfBatched with lda=32 — same pattern as n>=32.
    std::vector<LongType> paddedShape;
    if (batchSize > 1)
      paddedShape = {batchSize, static_cast<LongType>(lda), static_cast<LongType>(lda)};
    else
      paddedShape = {static_cast<LongType>(lda), static_cast<LongType>(lda)};

    auto paddedArr = NDArrayFactory::create_('c', paddedShape, input->dataType(), context);
    paddedArr->nullify();  // zero-fill

    // Copy each n×n matrix into the top-left corner of each lda×lda padded matrix using kernel.
    // Both arrays are C-order contiguous, so we know the exact layout without TAD:
    //   tempOutput: batch stride = n*n, row stride = n
    //   paddedArr:  batch stride = lda*lda, row stride = lda
    auto tempBuf = reinterpret_cast<F*>(tempOutput->specialBuffer());
    auto paddedBuf = reinterpret_cast<F*>(paddedArr->specialBuffer());
    dim3 copyDims((n * n) / 256 + 1, 256, 256);
    for (LongType i = 0; i < batchSize; i++) {
      copyPaddedBatch<F><<<copyDims.x, copyDims.y, copyDims.z, *stream>>>(
          tempBuf, static_cast<LongType>(n * n), static_cast<LongType>(n),
          paddedBuf, static_cast<LongType>(lda * lda), static_cast<LongType>(lda),
          i, n);
    }
    sd::DebugHelper::checkErrorCode(stream, "copyPaddedBatch (to padded) failed");


    // Build batch pointer array on host and copy to device.
    // Each pointer = paddedBuf + batch * lda * lda (contiguous C-order matrices).
    std::vector<F*> hostPtrs(batchSize);
    for (LongType i = 0; i < batchSize; i++) {
      hostPtrs[i] = paddedBuf + i * lda * lda;
    }
    F **paddedBatch = reinterpret_cast<F**>(sd::memory::CudaMemoryPool::getInstance().allocate(
        sizeof(F*) * batchSize, cholDevId, *stream));
    if (paddedBatch == nullptr) {
      delete paddedArr;
      THROW_EXCEPTION("helpers::cholesky_: Cannot allocate padded batch pointers");
    }
    cudaMemcpyAsync(paddedBatch, hostPtrs.data(), sizeof(F*) * batchSize,
                    cudaMemcpyHostToDevice, *stream);

    if (input->dataType() == DOUBLE)
      status = cusolverDnDpotrfBatched(handle, uplo, n, reinterpret_cast<double**>(paddedBatch), lda, dInfoArray, batchSize);
    else
      status = cusolverDnSpotrfBatched(handle, uplo, n, reinterpret_cast<float**>(paddedBatch), lda, dInfoArray, batchSize);

    if (CUSOLVER_STATUS_SUCCESS != status) {
      sd::memory::CudaMemoryPool::getInstance().free(paddedBatch, cholDevId, *stream);
      delete paddedArr;
      { std::string msg = "helpers::cholesky_: Cholesky factorization failed for batch; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
    }


    // Copy results back: padded lda×lda -> original n×n using kernel
    for (LongType i = 0; i < batchSize; i++) {
      copyPaddedBatch<F><<<copyDims.x, copyDims.y, copyDims.z, *stream>>>(
          paddedBuf, static_cast<LongType>(lda * lda), static_cast<LongType>(lda),
          tempBuf, static_cast<LongType>(n * n), static_cast<LongType>(n),
          i, n);
    }
    sd::DebugHelper::checkErrorCode(stream, "copyPaddedBatch (from padded) failed");

    sd::memory::CudaMemoryPool::getInstance().free(paddedBatch, cholDevId, *stream);
    delete paddedArr;
  } else {
    // n >= 32: no padding needed, use original batch pointers directly
    if (input->dataType() == DOUBLE)
      status = cusolverDnDpotrfBatched(handle, uplo, n, (double **)dArrayBatch, n, dInfoArray, batchSize);
    else
      status = cusolverDnSpotrfBatched(handle, uplo, n, (float **)dArrayBatch, n, dInfoArray, batchSize);

    if (CUSOLVER_STATUS_SUCCESS != status) {
      { std::string msg = "helpers::cholesky_: Cholesky factorization failed for batch; Error code: [" + std::to_string(status) + "]"; THROW_EXCEPTION(msg.c_str()); }
    }

  }

  // Build batch offsets for adjustResultsKernel: each matrix is n2 elements apart.
  std::vector<LongType> hostOffsets(batchSize);
  for (LongType i = 0; i < batchSize; i++) {
    hostOffsets[i] = i * n2;
  }
  LongType *devOffsets = reinterpret_cast<LongType*>(
      sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(LongType) * batchSize, cholDevId, *stream));
  cudaMemcpyAsync(devOffsets, hostOffsets.data(), sizeof(LongType) * batchSize,
                  cudaMemcpyHostToDevice, *stream);

  adjustResultsKernel<F><<<batchSize, n2, 128, *stream>>>(reinterpret_cast<F *>(tempOutput->specialBuffer()),
                                                          tempOutput->specialShapeInfo(), devOffsets, batchSize,
                                                          n);
  sd::DebugHelper::checkErrorCode(stream, "adjustResultsKernel failed");
  sd::memory::CudaMemoryPool::getInstance().free(devOffsets, cholDevId, *stream);


  sd::memory::CudaMemoryPool::getInstance().free(dArrayBatch, cholDevId, *stream);
  sd::memory::CudaMemoryPool::getInstance().free(dInfoArray, cholDevId, *stream);

  // Sync stream before assign — cuSOLVER and copy kernels ran on *stream,
  // but assign may use a different stream, so ensure results are visible
  cudaStreamSynchronize(*stream);

  if (!inplace)
    output->assign(tempOutput);
  else
    input->assign(tempOutput);

  delete tempOutput;
  NDArray::registerSpecialUse({output}, {input});
  cusolverDnDestroy(handle);
  return Status::OK;
}

//    template <typename T>
Status cholesky_(LaunchContext *context, NDArray *input, NDArray *output, bool inplace) {
  NDArray::prepareSpecialUse({output}, {input});
  if (input->dataType() == DOUBLE)
    cholesky__<double>(context, input, output, inplace);
  else if (input->dataType() == FLOAT32)
    cholesky__<float>(context, input, output, inplace);
  else {
    auto* shapePtr = input->getShapeAsVector();
    std::vector<sd::LongType> shape = *shapePtr;
    delete shapePtr;
    std::unique_ptr<NDArray> tempOutput(NDArrayFactory::create_('c', shape, FLOAT32, context));
    tempOutput->assign(input);
    cholesky__<float>(context, tempOutput.get(), tempOutput.get(), true);
    output->assign(tempOutput.get());
  }
  NDArray::registerSpecialUse({output}, {input});
  return Status::OK;
}

Status cholesky(LaunchContext *context, NDArray *input, NDArray *output, bool inplace) {
  return cholesky_(context, input, output, inplace);
}

BUILD_SINGLE_TEMPLATE( sd::Status inverse_, (sd::LaunchContext * context, NDArray *input, NDArray *output),
                      SD_FLOAT_NATIVE);

template <typename T>
SD_KERNEL SD_INLINE void logDetKernel(const T *inputBuf, const LongType *inputShape, LongType batchNum, const LongType *tadShape,
                            const LongType *tadOffsets, T *outputBuf, const LongType *outputShape) {
  __shared__ int n;
  if (threadIdx.x == 0) {
    n = shape::sizeAt(inputShape, -1);
  }
  __syncthreads();

  auto output = outputBuf;
  auto input = inputBuf;

  for (auto i = blockIdx.x; i < batchNum; i += gridDim.x) {
    auto current = input + tadOffsets[i];

    LongType zIndex;
    COORDS2INDEX(1, shape::stride(outputShape), &i, zIndex);
    for (auto e = threadIdx.x; e < n; e += blockDim.x) {
      LongType diag[] = {e, e};
      LongType xIndex;
      COORDS2INDEX(shape::rank(tadShape), shape::stride(tadShape), diag, xIndex);
      math::atomics::sd_atomicAdd(&output[zIndex], math::sd_log<T, T>(current[xIndex] * current[xIndex]));
    }
  }
}
// Explicit template instantiations for logDetKernel
template SD_KERNEL SD_INLINE void logDetKernel<float>(const float *inputBuf, const LongType *inputShape, LongType batchNum, const LongType *tadShape, const LongType *tadOffsets, float *outputBuf, const LongType *outputShape);
template SD_KERNEL SD_INLINE void logDetKernel<double>(const double *inputBuf, const LongType *inputShape, LongType batchNum, const LongType *tadShape, const LongType *tadOffsets, double *outputBuf, const LongType *outputShape);
template SD_KERNEL SD_INLINE void logDetKernel<float16>(const float16 *inputBuf, const LongType *inputShape, LongType batchNum, const LongType *tadShape, const LongType *tadOffsets, float16 *outputBuf, const LongType *outputShape);

template <typename T>
Status logdetFunctor_(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  auto n2 = input->sizeAt(-1) * input->sizeAt(-2);
  auto stream = context->getCudaStream();
  NDArray tempOutput(*input);

  cholesky(context, input, &tempOutput, false);

  auto outputBuf = output->dataBuffer()->template specialAsT<T>();
  auto inputBuf = tempOutput.dataBuffer()->template specialAsT<T>();
  output->nullify();

  std::vector<LongType> dims = {tempOutput.rankOf() - 2, tempOutput.rankOf() - 1};
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(tempOutput.shapeInfo(), &dims);
  logDetKernel<T><<<128, 512, 256, *stream>>>(inputBuf, tempOutput.specialShapeInfo(), packX->numberOfTads(),
                                              packX->specialShapeInfo(), packX->specialOffsets(), outputBuf,
                                              output->specialShapeInfo());
  sd::DebugHelper::checkErrorCode(stream, "logDetKernel failed");

  NDArray::registerSpecialUse({output}, {input});
  return Status::OK;
}
BUILD_SINGLE_TEMPLATE(Status logdetFunctor_, (LaunchContext *context, NDArray *input, NDArray *output), SD_FLOAT_NATIVE);

Status logdetFunctor(LaunchContext *context, NDArray *input, NDArray *output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), return logdetFunctor_, (context, input, output), SD_FLOAT_NATIVE);
}

/*
 * lup - batched input, batched outputs
 * */
Status lup(LaunchContext *context, NDArray *input, NDArray *compound, NDArray *permutation) {
  // input is read+written in-place by cuSolver; compound+permutation are outputs.
  NDArray::prepareSpecialUse({input, compound, permutation}, {input});
  BUILD_DOUBLE_SELECTOR(input->dataType(), permutation->dataType(), lup_, (context, input, compound, permutation),
                        SD_FLOAT_NATIVE, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({input, compound, permutation}, {});
  return Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
