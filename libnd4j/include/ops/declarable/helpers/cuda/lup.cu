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
#include <exceptions/cuda_exception.h>
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
static SD_KERNEL void invertKernelLow(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
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
static SD_KERNEL void upvertKernel(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
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
static SD_KERNEL void upvertKernelUp(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
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
static SD_KERNEL void invertLowKernel(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
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
static SD_KERNEL void invertUpKernel(void *invertedBuf, const LongType *invertedShape, const void *inputBuf,
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
  NDArray::prepareSpecialUse({invertedMatrix}, {inputMatrix});
}

// ------------------------------------------------------------------------------------------------------------------ //
// determinant kernel - accumulation product of all values on the main diagonal
template <typename T>
static SD_KERNEL void determinantKernel(T *compound, T *result, LongType len) {
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
static SD_KERNEL void determinantLogKernel(T *compound, T *result, LongType len) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  for (auto i = start; i < len; i += step) {
    auto pos = i * len + i;
    // sum logs of all diagonal elements
    math::atomics::sd_atomicAdd(result, math::sd_log<T, T>(math::sd_abs<T,T>(compound[pos])));
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// kernel to copy matrix with given shape to compound tensor with given pos
// output - a N-D tensor buffer with rank not less than 2, input - 2D square n x n matrix with n = rowLen
template <typename T, typename F>
static SD_KERNEL void fillMatrix(void *output, const LongType *outShape, const void *input, const LongType *inputShape,
                                 LongType pos, LongType rowLen) {
  // Shared memory caching for rank, shape, and stride
  __shared__ F *matrix;
  __shared__ const T *inputBuf;
  __shared__ LongType inputLen;
  __shared__ LongType n2;
  __shared__ LongType inputRank;
  __shared__ const LongType *inputShapePtr;
  __shared__ const LongType *inputStridePtr;

  if (threadIdx.x == 0) {
    matrix = reinterpret_cast<F *>(output);
    inputBuf = reinterpret_cast<const T *>(input);
    inputLen = shape::length(inputShape);
    n2 = rowLen * rowLen;
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int k = pos + start, j = start; j < n2; k += step, j += step) {
    LongType coords[SD_MAX_RANK];
    LongType xIndex;

    // Use cached rank, shape, and stride
    INDEX2COORDS(k, inputRank, inputShapePtr, coords);
    COORDS2INDEX(inputRank, inputStridePtr, coords, xIndex);

    matrix[j] = static_cast<F>(inputBuf[xIndex]);
  }
}


// ------------------------------------------------------------------------------------------------------------------ //
// same as above, but without type conversion
template <typename T>
static SD_KERNEL void returnMatrix(void *output, const LongType *outputShape, const void *input,
                                   const LongType *inputShape, LongType pos, LongType rowLen) {
  // Shared memory caching for rank, shape, and stride
  __shared__ LongType outputLen;
  __shared__ LongType n2;
  __shared__ LongType outputRank;
  __shared__ const LongType *outputShapePtr;
  __shared__ const LongType *outputStridePtr;

  auto matrix = reinterpret_cast<const T *>(input);
  auto outputBuf = reinterpret_cast<T *>(output);

  if (threadIdx.x == 0) {
    outputLen = shape::length(inputShape);
    n2 = rowLen * rowLen;
    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int k = pos + start, j = start; j < n2; k += step, j += step) {
    LongType zCoords[SD_MAX_RANK];
    LongType zIndex;

    // Use cached rank, shape, and stride
    INDEX2COORDS(k, outputRank, outputShapePtr, zCoords);
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zIndex);

    outputBuf[zIndex] = matrix[j];
  }
}


// ------------------------------------------------------------------------------------------------------------------ //
// fill up permutaion matrix kernel. Permutation matrix filled with zeros and ones
template <typename F>
static SD_KERNEL void fillUpPermutation(void *output, const LongType *shape, int *source, int rowNum) {
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
    throw cuda_exception::build("Cannot set up stream for cuda solver", status);
  }
  int lwork = 0;
  int *d_info = nullptr;
  // allocate memory for permutation vector
  auto err = cudaMalloc((void **)&d_info, sizeof(LongType));
  if (err) {
    throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver info buffer", err);
  }

  DataType dtype = input->dataType();
  switch (dtype) {  // there are two implementations with cublas for LUP decomposition - double and float

    case DOUBLE: {
      double *d_work = nullptr;
      // compute internal buffer size
      double *matrix = reinterpret_cast<double *>(input->specialBuffer());
      status = cusolverDnDgetrf_bufferSize(*cusolverH, n, n, matrix, n, &lwork);
      if (CUSOLVER_STATUS_SUCCESS != status) {
        throw cuda_exception::build("helpers::lup_: Cannot create cuSolver handle", status);
      }

      err = cudaMalloc((void **)&d_work, sizeof(float) * lwork);
      if (err) {
        throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver data buffer", err);
      }

      if (permutation == nullptr) {
        status = cusolverDnDgetrf(*cusolverH, n, n, matrix, n, d_work, nullptr, d_info);

        if (status != CUSOLVER_STATUS_SUCCESS) {
          throw cuda_exception::build("helpers::lup_: LU factorization is failed due ", status);
        }
      } else {
        std::vector<LongType> shape = {n};
        NDArray permutVector('c', shape, INT32, context);
        int *permutationBuf = permutVector.dataBuffer()->specialAsT<int>();
        status = cusolverDnDgetrf(*cusolverH, n, n, matrix, n, d_work, permutationBuf, d_info);
        if (status != CUSOLVER_STATUS_SUCCESS) {
          throw cuda_exception::build("helpers::lup_: LU factorization is failed due ", status);
        }

        if (permutation->rankOf() == 2) {
          fillUpPermutation<double><<<n, n, 1024, *stream>>>(permutation->specialBuffer(),
                                                             permutation->specialShapeInfo(), permutationBuf, n);
          sd::DebugHelper::checkErrorCode(stream, "fillUpPermutation failed");

        } else {
          permutVector.tickWriteDevice();
          input->tickWriteDevice();
          compound->assign(input);
          permutation->assign(&permutVector);
        }
      }
      err = cudaFree(d_work);
      if (err) {
        throw cuda_exception::build("helpers::lup_: Cannot deallocate memory for solver data buffer", err);
      }
    } break;
    case FLOAT32: {
      float *matrix = reinterpret_cast<float *>(input->specialBuffer());
      float *d_work = nullptr;

      status = cusolverDnSgetrf_bufferSize(*cusolverH, n, n, matrix, n, &lwork);
      if (CUSOLVER_STATUS_SUCCESS != status) {
        throw cuda_exception::build("helpers::lup_: Cannot create cuSolver handle", status);
      }

      err = cudaMalloc((void **)&d_work, sizeof(float) * lwork);
      if (err) {
        throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver data buffer", err);
      }

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

          permutation->tickWriteDevice();
        } else {
          input->tickWriteDevice();
          compound->assign(input);
          permutation->assign(&permutVector);
        }
      }
      err = cudaFree(d_work);
      if (err) {
        throw cuda_exception::build("helpers::lup_: Cannot deallocate memory for solver data buffer", err);
      }
    }
  }
  if (CUSOLVER_STATUS_SUCCESS != status) {
    throw cuda_exception::build("helpers::lup_: Cannot make LU decomposition", status);
  }
  err = cudaFree(d_info);
  if (err) {
    throw cuda_exception::build("helpers::lup_: Cannot deallocate memory for solver info buffer", err);
  }

  input->tickWriteDevice();
}
// ------------------------------------------------------------------------------------------------------------------ //

BUILD_DOUBLE_TEMPLATE(template void lup_,
                      (LaunchContext * context, NDArray *input, NDArray *output, NDArray *permutation), SD_FLOAT_NATIVE,
                      SD_INDEXING_TYPES);

template <typename T>
static void swapRows_(NDArray *matrix, LongType theFirst, LongType theSecond) {
  if (theFirst != theSecond)
    for (LongType i = 0; i < matrix->columns(); i++) {
      math::sd_swap(matrix->r<T>(theFirst, i), matrix->r<T>(theSecond, i));
    }
}
BUILD_SINGLE_TEMPLATE(template void swapRows_, (NDArray * matrix, sd::LongType theFirst, sd::LongType theSecond),
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
      compound->r<T>(i, k) = input.t<T>(i, k) - sum;
    }

    // Lower Triangular
    for (LongType k = i + 1; k < rowNum; k++) {
      // Summation of L(k, j) * U(j, i)
      LongType sum = 0;
      for (LongType j = 0; j < i; j++) sum += compound->t<T>(k, j) * compound->t<T>(j, i);

      // Evaluating L(k, i)
      compound->r<T>(k, i) = (input.t<T>(k, i) - sum) / compound->t<T>(i, i);
    }
  }
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
      maxValue = sd::math::sd_max(maxValue, sd::math::sd_abs<T,T>(compoundBuffer[xIndex]));
      result = rowCounter;
    }
  }

  return result;
}

template <typename T, typename I>
static void luNN_(LaunchContext *context, NDArray *compound, NDArray *permutation, LongType rowNum) {
  NDArray::preparePrimaryUse({compound}, {permutation});

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
  } else {  // Doolittle algorithm with LU decomposition
    doolitleLU<T>(context, compound, rowNum);
  }

  NDArray::registerPrimaryUse({compound}, {permutation});
}


template <typename T, typename I>
static void lu_(LaunchContext *context, NDArray *input, NDArray *output, NDArray *permutationVectors) {
  NDArray::preparePrimaryUse({output}, {input, permutationVectors});

  auto n = input->sizeAt(-1);

  output->assign(input);  // fill up output tensor with zeros
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
  std::vector<LongType> dims();
  std::vector<LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};

  auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, DataTypeUtils::fromT<T>(), context);
  auto det = NDArrayFactory::create<T>(1, context);
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input});
  dim3 launchDims = getLaunchDims("logAbsDeterminant");
  float one = 1.f;
  output->assign(one);

  // Cache rank, shape, and stride outside the loop
  sd::LongType outputRank = shape::rank(output->shapeInfo());
  const sd::LongType* outputShape = shape::shapeOf(output->shapeInfo());
  const sd::LongType* outputStride = shape::stride(output->shapeInfo());

  for (int e = 0; e < output->lengthOf(); e++) {
    LongType pos = e * n2;

    // Fill matrix using the CUDA kernel
    fillMatrix<T, T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), pos, n);
    sd::DebugHelper::checkErrorCode(stream, "fillMatrix failed");

    // Perform LU decomposition
    lup_<T, int>(context, &matrix, nullptr, nullptr);

    // Precompute coordinates and offsets
    LongType offsetCoords[SD_MAX_RANK];
    LongType offset;
    INDEX2COORDS(e, outputRank, outputShape, offsetCoords);
    COORDS2INDEX(outputRank, outputStride, offsetCoords, offset);

    // Execute determinant kernel
    auto inputBuf = reinterpret_cast<T*>(matrix.specialBuffer());
    auto outputBuf = reinterpret_cast<T*>(output->specialBuffer()) + offset;
    determinantKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuf, outputBuf, n);
    sd::DebugHelper::checkErrorCode(stream, "determinantKernel failed");
  }

  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}


Status determinant(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), return determinant_, (context, input, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
Status logAbsDeterminant_(LaunchContext *context, NDArray *input, NDArray *output) {
  LongType n = input->sizeAt(-1);
  LongType n2 = n * n;
  std::vector<LongType> dims();
  std::vector<LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};
  DataType dtype = input->dataType();
  if (dtype != DOUBLE) dtype = FLOAT32;

  auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, dtype, context);
  auto det = NDArrayFactory::create<T>(1, context);
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input});
  dim3 launchDims = getLaunchDims("logAbsDeterminant");
  float zero = 0.f;
  output->assign(zero);

  // Cache rank, shape, and stride outside the loop
  sd::LongType outputRank = shape::rank(output->shapeInfo());
  const sd::LongType* outputShape = shape::shapeOf(output->shapeInfo());
  const sd::LongType* outputStride = shape::stride(output->shapeInfo());

  for (int e = 0; e < output->lengthOf(); e++) {
    LongType pos = e * n2;
    fillMatrix<T, T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), pos, n);
    sd::DebugHelper::checkErrorCode(stream, "fillMatrix failed");

    lup_<T, int>(context, &matrix, nullptr, nullptr);

    // Precompute coordinates and offsets
    LongType offsetCoords[SD_MAX_RANK];
    LongType offset;
    INDEX2COORDS(e, outputRank, outputShape, offsetCoords);
    COORDS2INDEX(outputRank, outputStride, offsetCoords, offset);

    auto inputBuf = reinterpret_cast<T *>(matrix.specialBuffer());
    auto outputBuf = reinterpret_cast<T *>(output->specialBuffer()) + offset;
    determinantLogKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuf, outputBuf, n);
    sd::DebugHelper::checkErrorCode(stream, "determinantLogKernel failed");
  }

  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}


Status logAbsDeterminant(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), return logAbsDeterminant_, (context, input, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
static SD_KERNEL void fillLowerUpperKernel(void *lowerBuf, const LongType *lowerShape, void *upperBuf,
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

  NDArray matrix = NDArrayFactory::create('c', {n, n}, dtype, context);
  NDArray upper = NDArrayFactory::create('c', {n, n}, dtype, context);
  NDArray lower = NDArrayFactory::create('c', {n, n}, dtype, context);
  NDArray compound = NDArrayFactory::create('c', {n, n}, dtype, context);
  NDArray permutation = NDArrayFactory::create('c', {n, n}, dtype, context);

  std::vector<LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};
  std::vector<LongType> dims3 = {output->rankOf() - 2, output->rankOf() - 1};

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &dims2);

  auto stream = context->getCudaStream();

  for (auto i = 0LL; i < packX->numberOfTads(); i++) {
    fillMatrix<T, T><<<1, n2, 1024, *stream>>>(matrix.specialBuffer(), matrix.specialShapeInfo(),
                                               input->specialBuffer(), input->specialShapeInfo(), i * n2, n);
    sd::DebugHelper::checkErrorCode(stream, "fillMatrix failed");
    matrix.tickWriteDevice();
    lup_<T, int>(context, &matrix, nullptr, nullptr);
    fillLowerUpperKernel<T><<<n, n, 1024, *stream>>>(lower.specialBuffer(), lower.specialShapeInfo(),
                                                     upper.specialBuffer(), upper.specialShapeInfo(),
                                                     matrix.specialBuffer(), matrix.specialShapeInfo(), n);
    sd::DebugHelper::checkErrorCode(stream, "fillLowerUpperKernel failed");

    lower.tickWriteDevice();
    upper.tickWriteDevice();
    int zero = 0;
    matrix.assign(zero);
    invertUpperMatrix(context, &upper, &matrix);  // U^{-1}
    matrix.tickWriteDevice();
    compound.assign(zero);
    invertLowerMatrix(context, &lower, &compound);  // L{-1}
    compound.tickWriteDevice();

    MmulHelper::mmul(&matrix, &compound, &upper, 1.0, 0.0);
    upper.tickWriteDevice();
    returnMatrix<T><<<1, n2, 1024, *stream>>>(output->specialBuffer(), output->specialShapeInfo(),
                                              upper.specialBuffer(), upper.specialShapeInfo(), i * n2, n);
    sd::DebugHelper::checkErrorCode(stream, "returnMatrix failed");
  }
  return Status::OK;
}

Status inverse(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), return inverse_, (context, input, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {input});
}

bool checkCholeskyInput(LaunchContext *context, NDArray *input) { return true; }

template <typename F>
SD_KERNEL void fillBatchKernel(F **dArrayBatch, F *buf, const LongType *offsets, LongType batchSize) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (auto i = start; i < batchSize; i += step) {
    dArrayBatch[i] = buf + offsets[i];
  }
}

template <typename F>
SD_KERNEL void adjustResultsKernel(F *dArray, const LongType *shape, const LongType *offsets, LongType batchSize,
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
    throw cuda_exception::build("helpers::cholesky_: Cannot create solver handle", status);
  }
  F **dArrayBatch = nullptr;
  std::vector<LongType> dims = {tempOutput.rankOf() - 2, tempOutput.rankOf() - 1};
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(tempOutput.shapeInfo(), &dims);
  const LongType batchSize = packX->numberOfTads();
  int *dInfoArray = nullptr;
  auto err = cudaMalloc((void **)&dArrayBatch, sizeof(F *) * batchSize);
  if (err) {
    throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver batch data buffer", err);
  }
  err = cudaMalloc((void **)&dInfoArray, sizeof(LongType) * batchSize);
  if (err) {
    throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver errors buffer", err);
  }
  auto stream = context->getCudaStream();
  fillBatchKernel<F><<<1, batchSize, 128, *stream>>>(dArrayBatch, reinterpret_cast<F *>(tempOutput.specialBuffer()),
                                                     packX->specialOffsets(), batchSize);
  sd::DebugHelper::checkErrorCode(stream, "fillBatchKernel failed");

  status = cusolverDnSetStream(handle, *stream);
  if (CUSOLVER_STATUS_SUCCESS != status) {
    throw cuda_exception::build("helpers::cholesky_: Cannot set stream to solver handle", status);
  }
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  if (input->dataType() == DOUBLE)
    status = cusolverDnDpotrfBatched(handle, uplo, n, (double **)dArrayBatch, n, dInfoArray, batchSize);
  else
    status = cusolverDnSpotrfBatched(handle, uplo, n, (float **)dArrayBatch, n, dInfoArray, batchSize);

  if (CUSOLVER_STATUS_SUCCESS != status) {
    throw cuda_exception::build("helpers::cholesky_: Cholesky factorization failed for batch", status);
  }
  adjustResultsKernel<F><<<batchSize, n2, 128, *stream>>>(reinterpret_cast<F *>(tempOutput.specialBuffer()),
                                                          packX->specialShapeInfo(), packX->specialOffsets(), batchSize,
                                                          n);
  sd::DebugHelper::checkErrorCode(stream, "adjustResultsKernel failed");

  err = cudaFree(dArrayBatch);
  if (err) {
    throw cuda_exception::build("helpers::cholesky_: Cannot deallocate memory for solver batch data buffer", err);
  }
  err = cudaFree(dInfoArray);
  if (err) {
    throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver errors buffer", err);
  }

  if (!inplace)
    output->assign(&tempOutput);
  else
    input->assign(&tempOutput);

  NDArray::registerSpecialUse({output}, {input});
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
    std::vector<sd::LongType> shape = input->getShapeAsVector();
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

BUILD_SINGLE_TEMPLATE(template sd::Status inverse_, (sd::LaunchContext * context, NDArray *input, NDArray *output),
                      SD_FLOAT_NATIVE);

template <typename T>
SD_KERNEL void logDetKernel(const T *inputBuf, const LongType *inputShape, LongType batchNum, const LongType *tadShape,
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
template <typename T>
Status logdetFunctor_(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  auto n2 = input->sizeAt(-1) * input->sizeAt(-2);
  auto stream = context->getCudaStream();
  NDArray tempOutput(*input);

  cholesky(context, input, &tempOutput, false);

  auto outputBuf = output->dataBuffer()->specialAsT<T>();
  auto inputBuf = tempOutput.dataBuffer()->specialAsT<T>();
  output->nullify();

  std::vector<LongType> dims = {tempOutput.rankOf() - 2, tempOutput.rankOf() - 1};
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(tempOutput.shapeInfo(), &dims);
  logDetKernel<T><<<128, 512, 256, *stream>>>(inputBuf, tempOutput.specialShapeInfo(), packX->numberOfTads(),
                                              packX->specialShapeInfo(), packX->specialOffsets(), outputBuf,
                                              output->specialShapeInfo());
  sd::DebugHelper::checkErrorCode(stream, "logDetKernel failed");

  output->tickWriteDevice();
  NDArray::registerSpecialUse({output}, {input});
  return Status::OK;
}

Status logdetFunctor(LaunchContext *context, NDArray *input, NDArray *output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), return logdetFunctor_, (context, input, output), SD_FLOAT_NATIVE);
}

/*
 * lup - batched input, batched outputs
 * */
Status lup(LaunchContext *context, NDArray *input, NDArray *compound, NDArray *permutation) {
  BUILD_DOUBLE_SELECTOR(input->dataType(), permutation->dataType(), lup_, (context, input, compound, permutation),
                        SD_FLOAT_NATIVE, SD_INDEXING_TYPES);
  return Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
