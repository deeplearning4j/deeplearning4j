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
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/op_types.h>
#include <ops/declarable/helpers/top_k.h>

#include "execution/Threads.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

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
#if defined(HAVE_ZLUDA)
  THROW_EXCEPTION("LUP factorization requires cuSolver and is not supported by the ZLUDA backend");
#else
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
          dim3 permutationDims = getLaunchDims("lup");
          fillUpPermutation<double><<<permutationDims.x, permutationDims.y, 0, *stream>>>(
              permutation->specialBuffer(), permutation->specialShapeInfo(), permutationBuf, n);
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
          dim3 permutationDims = getLaunchDims("lup");
          fillUpPermutation<I><<<permutationDims.x, permutationDims.y, 0, *stream>>>(
              permutation->specialBuffer(), permutation->specialShapeInfo(), permutationBuf, n);
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
#endif
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
#if !defined(HAVE_ZLUDA)
// cuSOLVER's LU factorization (getrf) and solve (getrs) for the two types it factorizes
static cusolverStatus_t getrfBufferSize(cusolverDnHandle_t handle, int n, float *a, int *lwork) {
  return cusolverDnSgetrf_bufferSize(handle, n, n, a, n, lwork);
}
static cusolverStatus_t getrfBufferSize(cusolverDnHandle_t handle, int n, double *a, int *lwork) {
  return cusolverDnDgetrf_bufferSize(handle, n, n, a, n, lwork);
}
static cusolverStatus_t getrf(cusolverDnHandle_t handle, int n, float *a, float *work, int *pivots, int *info) {
  return cusolverDnSgetrf(handle, n, n, a, n, work, pivots, info);
}
static cusolverStatus_t getrf(cusolverDnHandle_t handle, int n, double *a, double *work, int *pivots, int *info) {
  return cusolverDnDgetrf(handle, n, n, a, n, work, pivots, info);
}
static cusolverStatus_t getrs(cusolverDnHandle_t handle, int n, const float *a, const int *pivots, float *b,
                              int *info) {
  return cusolverDnSgetrs(handle, CUBLAS_OP_N, n, n, a, n, pivots, b, n, info);
}
static cusolverStatus_t getrs(cusolverDnHandle_t handle, int n, const double *a, const int *pivots, double *b,
                              int *info) {
  return cusolverDnDgetrs(handle, CUBLAS_OP_N, n, n, a, n, pivots, b, n, info);
}
#endif

// ------------------------------------------------------------------------------------------------------------------ //
// LU-factorizes the n x n matrices of a C-order batch in place with partial pivoting: the pivots of matrix m go to
// pivots + m * n (1-based rows) and getrf's info to infos[m] (k > 0: U's k-th pivot is zero). cuSOLVER works on
// column-major matrices, so it sees a row-major matrix A as A^T: it factorizes P A^T = L U, and det A = det A^T.
template <typename F>
static void luBatch_(LaunchContext *context, F *a, int *pivots, int *infos, LongType n, LongType batch) {
#if defined(HAVE_ZLUDA)
  THROW_EXCEPTION("LU factorization requires cuSolver and is not supported by the ZLUDA backend");
#else
  auto stream = context->getCudaStream();
  std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());
  auto handle = reinterpret_cast<cusolverDnHandle_t *>(context->getCusolverHandle());
  auto status = cusolverDnSetStream(*handle, *stream);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    std::string msg = "helpers::luBatch_: Cannot set up stream for cuda solver; Error code: [" + std::to_string(status) + "]";
    THROW_EXCEPTION(msg.c_str());
  }
  const int dim = static_cast<int>(n);
  int lwork = 0;
  status = getrfBufferSize(*handle, dim, a, &lwork);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    std::string msg = "helpers::luBatch_: Cannot size the LU workspace; Error code: [" + std::to_string(status) + "]";
    THROW_EXCEPTION(msg.c_str());
  }
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  auto &pool = sd::memory::CudaMemoryPool::getInstance();
  auto work = reinterpret_cast<F *>(pool.allocate(sizeof(F) * (lwork > 0 ? lwork : 1), deviceId, *stream));
  if (work == nullptr) THROW_EXCEPTION("helpers::luBatch_: Cannot allocate the LU workspace");
  for (LongType m = 0; m < batch; m++) {
    status = getrf(*handle, dim, a + m * n * n, work, pivots + m * n, infos + m);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      std::string msg = "helpers::luBatch_: LU factorization failed; Error code: [" + std::to_string(status) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
  }
  pool.free(work, deviceId, *stream);
#endif
}

// ------------------------------------------------------------------------------------------------------------------ //
// det A = (-1)^(row exchanges) * prod U_ii and log |det A| = sum log |U_ii| from luBatch_'s factors (det A^T = det A);
// each output element is reached through the output's own strides
template <typename F, typename T>
static SD_KERNEL void determinantFromLuKernel(const F *factors, const int *pivots, LongType n, LongType batch,
                                              bool logAbs, T *output, const LongType *outputShape) {
  const int outputRank = shape::rank(outputShape);
  const LongType *outputShapeOf = shape::shapeOf(outputShape);
  const LongType *outputStride = shape::stride(outputShape);
  LongType zCoords[SD_MAX_RANK];

  for (LongType m = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; m < batch;
       m += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const F *lu = factors + m * n * n;
    const int *p = pivots + m * n;
    F value = logAbs ? static_cast<F>(0) : static_cast<F>(1);
    bool negative = false, singular = false;
    for (LongType e = 0; e < n; e++) {
      const F u = lu[e * n + e];
      if (u == static_cast<F>(0)) singular = true;
      if (logAbs)
        value += math::sd_log<F, F>(math::sd_abs<F, F>(u));
      else
        value *= u;
      // the pivots are 1-based rows: a pivot other than row e + 1 is a row exchange
      if (p[e] != e + 1) negative = !negative;
    }
    if (!logAbs && negative) value = -value;
    // a zero pivot is a singular matrix: log |det| = -inf (sd_log takes log 0 as log epsilon)
    if (logAbs && singular) value = -DataTypeUtils::infOrMax<F>();

    LongType zOffset;
    INDEX2COORDS(m, outputRank, outputShapeOf, zCoords);
    COORDS2INDEX(outputRank, outputStride, zCoords, zOffset);
    output[zOffset] = static_cast<T>(value);
  }
}

template <typename T>
static Status determinantFromLu_(LaunchContext *context, NDArray *input, NDArray *output, bool logAbs) {
  const LongType n = input->sizeAt(-1);
  const LongType batch = output->lengthOf();
  if (batch == 0) return Status::OK;
  if (n == 0) {  // the determinant of a 0 x 0 matrix is 1
    float value = logAbs ? 0.f : 1.f;
    output->assign(value);
    return Status::OK;
  }
  // cuSOLVER factorizes FLOAT32 and DOUBLE matrices: the other types are factorized in FLOAT32
  const DataType workType = DataTypeUtils::fromT<T>() == DOUBLE ? DOUBLE : FLOAT32;
  std::vector<LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());
  NDArray factors('c', shape, workType, context);
  factors.assign(input);

  auto stream = context->getCudaStream();
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  auto &pool = sd::memory::CudaMemoryPool::getInstance();
  auto pivots = reinterpret_cast<int *>(pool.allocate(sizeof(int) * n * batch, deviceId, *stream));
  auto infos = reinterpret_cast<int *>(pool.allocate(sizeof(int) * batch, deviceId, *stream));
  if (pivots == nullptr || infos == nullptr) THROW_EXCEPTION("helpers::determinant: Cannot allocate the solver buffers");

  NDArray::prepareSpecialUse({&factors}, {});
  if (workType == DOUBLE)
    luBatch_<double>(context, reinterpret_cast<double *>(factors.specialBuffer()), pivots, infos, n, batch);
  else
    luBatch_<float>(context, reinterpret_cast<float *>(factors.specialBuffer()), pivots, infos, n, batch);
  NDArray::registerSpecialUse({&factors}, {});

  NDArray::prepareSpecialUse({output}, {&factors});
  dim3 launchDims = getLaunchDims("logAbsDeterminant");
  auto outputBuf = reinterpret_cast<T *>(output->specialBuffer());
  if (workType == DOUBLE)
    determinantFromLuKernel<double, T><<<launchDims.x, launchDims.y, 0, *stream>>>(
        reinterpret_cast<const double *>(factors.specialBuffer()), pivots, n, batch, logAbs, outputBuf,
        output->specialShapeInfo());
  else
    determinantFromLuKernel<float, T><<<launchDims.x, launchDims.y, 0, *stream>>>(
        reinterpret_cast<const float *>(factors.specialBuffer()), pivots, n, batch, logAbs, outputBuf,
        output->specialShapeInfo());
  sd::DebugHelper::checkErrorCode(stream, "determinantFromLuKernel failed");
  NDArray::registerSpecialUse({output}, {&factors});

  pool.free(pivots, deviceId, *stream);
  pool.free(infos, deviceId, *stream);
  return Status::OK;
}

template <typename T>
static Status determinant_(LaunchContext *context, NDArray *input, NDArray *output) {
  return determinantFromLu_<T>(context, input, output, false);
}
BUILD_SINGLE_TEMPLATE(Status determinant_, (LaunchContext *context, NDArray *input, NDArray *output), SD_FLOAT_NATIVE);

Status determinant(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), return determinant_, (context, input, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
Status logAbsDeterminant_(LaunchContext *context, NDArray *input, NDArray *output) {
  return determinantFromLu_<T>(context, input, output, true);
}
BUILD_SINGLE_TEMPLATE(Status logAbsDeterminant_, (LaunchContext *context, NDArray *input, NDArray *output), SD_FLOAT_NATIVE);

Status logAbsDeterminant(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  BUILD_SINGLE_SELECTOR(input->dataType(), return logAbsDeterminant_, (context, input, output), SD_FLOAT_NATIVE);
  NDArray::registerSpecialUse({output}, {input});
}


// ------------------------------------------------------------------------------------------------------------------ //
// Inverts the n x n matrices of a C-order batch. On entry factors holds the matrices and inverses an identity matrix
// per matrix; on return factors holds their LU factors and inverses the inverses. cuSOLVER sees a row-major matrix A
// as A^T: solving A^T X = I with the factors of A^T gives X = (A^T)^-1 = (A^-1)^T, which read back row-major is A^-1.
template <typename F>
static Status invertBatch_(LaunchContext *context, NDArray *factors, NDArray *inverses, LongType n, LongType batch) {
#if defined(HAVE_ZLUDA)
  THROW_EXCEPTION("Matrix inversion requires cuSolver and is not supported by the ZLUDA backend");
  // THROW_EXCEPTION is not declared [[noreturn]], so MSVC requires this.
  return Status::OK;
#else
  auto stream = context->getCudaStream();
  auto a = reinterpret_cast<F *>(factors->specialBuffer());
  auto x = reinterpret_cast<F *>(inverses->specialBuffer());
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  auto &pool = sd::memory::CudaMemoryPool::getInstance();
  auto pivots = reinterpret_cast<int *>(pool.allocate(sizeof(int) * n * batch, deviceId, *stream));
  auto infos = reinterpret_cast<int *>(pool.allocate(sizeof(int) * batch, deviceId, *stream));
  if (pivots == nullptr || infos == nullptr) THROW_EXCEPTION("helpers::inverse: Cannot allocate the solver buffers");

  luBatch_<F>(context, a, pivots, infos, n, batch);

  // a zero pivot means the matrix is singular
  std::vector<int> zeroPivots(batch);
  cudaMemcpyAsync(zeroPivots.data(), infos, sizeof(int) * batch, cudaMemcpyDeviceToHost, *stream);
  cudaStreamSynchronize(*stream);
  Status result = Status::OK;
  for (LongType m = 0; m < batch; m++) {
    if (zeroPivots[m] > 0) {
      sd_printf("matrix_inverse: The matrix %i has no inverse: its LU factorization has a zero pivot.\n", (int)m);
      result = Status::VALIDATION;
      break;
    }
  }

  if (result == Status::OK) {
    std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());
    auto handle = reinterpret_cast<cusolverDnHandle_t *>(context->getCusolverHandle());
    auto status = cusolverDnSetStream(*handle, *stream);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      std::string msg = "helpers::inverse: Cannot set up stream for cuda solver; Error code: [" + std::to_string(status) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
    const int dim = static_cast<int>(n);
    for (LongType m = 0; m < batch; m++) {
      status = getrs(*handle, dim, a + m * n * n, pivots + m * n, x + m * n * n, infos + m);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::string msg = "helpers::inverse: Solving for the inverse failed; Error code: [" + std::to_string(status) + "]";
        THROW_EXCEPTION(msg.c_str());
      }
    }
  }

  pool.free(pivots, deviceId, *stream);
  pool.free(infos, deviceId, *stream);
  return result;
#endif
}

template <typename T>
static Status inverse_(LaunchContext *context, NDArray *input, NDArray *output) {
  const LongType n = input->sizeAt(-1);
  if (n == 0 || output->lengthOf() == 0) return Status::OK;
  const LongType batch = input->lengthOf() / (n * n);
  // cuSOLVER factorizes FLOAT32 and DOUBLE matrices: the other types are inverted in FLOAT32
  const DataType workType = DataTypeUtils::fromT<T>() == DOUBLE ? DOUBLE : FLOAT32;
  std::vector<LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());
  NDArray factors('c', shape, workType, context);
  factors.assign(input);

  // an identity matrix per matrix of the batch
  std::vector<LongType> matrixShape = {n, n};
  NDArray identity('c', matrixShape, workType, context);
  identity.setIdentity();
  NDArray inverses('c', shape, workType, context);
  NDArray::prepareSpecialUse({&inverses}, {&identity});
  auto stream = context->getCudaStream();
  const size_t matrixBytes = static_cast<size_t>(n * n) * DataTypeUtils::sizeOfElement(workType);
  for (LongType m = 0; m < batch; m++)
    cudaMemcpyAsync(static_cast<int8_t *>(inverses.specialBuffer()) + m * matrixBytes, identity.specialBuffer(),
                    matrixBytes, cudaMemcpyDeviceToDevice, *stream);
  NDArray::registerSpecialUse({&inverses}, {&identity});

  NDArray::prepareSpecialUse({&factors, &inverses}, {});
  const Status status = workType == DOUBLE ? invertBatch_<double>(context, &factors, &inverses, n, batch)
                                           : invertBatch_<float>(context, &factors, &inverses, n, batch);
  NDArray::registerSpecialUse({&factors, &inverses}, {});
  if (status != Status::OK) return status;

  output->assign(&inverses);
  // factors, identity and inverses are released when this returns: wait for the work that reads them
  PointersManager manager(context, "inverse");
  manager.synchronize();
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
#if defined(HAVE_ZLUDA)
  THROW_EXCEPTION("Cholesky factorization requires cuSolver and is not supported by the ZLUDA backend");
  // THROW_EXCEPTION is not declared [[noreturn]], so MSVC requires this.
  return Status::OK;
#else
  if (!inplace) output->assign(input);
  // the batch pointers below step n * n elements per matrix: the factorization works on a C-order copy
  auto tempOutput = output->dup('c');
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

  // the kernel strides over the matrices by block and over the rows by thread: any launch size covers them
  dim3 adjustDims = getLaunchDims("lup");
  adjustResultsKernel<F><<<adjustDims.x, adjustDims.y, 0, *stream>>>(reinterpret_cast<F *>(tempOutput->specialBuffer()),
                                                                     tempOutput->specialShapeInfo(), devOffsets,
                                                                     batchSize, n);
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
#endif
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
    NDArray *tempOutput = NDArrayFactory::create_('c', shape, FLOAT32, context);
    tempOutput->assign(input);
    cholesky__<float>(context, tempOutput, tempOutput, true);
    output->assign(tempOutput);
    delete tempOutput;
  }
  NDArray::registerSpecialUse({output}, {input});
  return Status::OK;
}

Status cholesky(LaunchContext *context, NDArray *input, NDArray *output, bool inplace) {
  return cholesky_(context, input, output, inplace);
}

BUILD_SINGLE_TEMPLATE( sd::Status inverse_, (sd::LaunchContext * context, NDArray *input, NDArray *output),
                      SD_FLOAT_NATIVE);

// log det A = sum over the diagonal of A's Cholesky factor L of log(L_ii^2). The factors are a C-order copy, n * n
// elements apart; each output element is reached through the output's own strides, whatever its rank.
template <typename T>
static SD_KERNEL void logDetKernel(const T *factors, LongType n, LongType batchNum, T *output,
                                   const LongType *outputShape) {
  using AccT = typename simdOps::AggregateType<T>::type;
  const int outputRank = shape::rank(outputShape);
  const LongType *outputShapeOf = shape::shapeOf(outputShape);
  const LongType *outputStride = shape::stride(outputShape);
  LongType zCoords[SD_MAX_RANK];

  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < batchNum;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const T *current = factors + i * n * n;
    AccT sum = static_cast<AccT>(0);
    for (LongType e = 0; e < n; e++) {
      const AccT diagonal = static_cast<AccT>(current[e * n + e]);
      sum += math::sd_log<AccT, AccT>(diagonal * diagonal);
    }
    LongType zOffset;
    INDEX2COORDS(i, outputRank, outputShapeOf, zCoords);
    COORDS2INDEX(outputRank, outputStride, zCoords, zOffset);
    output[zOffset] = static_cast<T>(sum);
  }
}

template <typename T>
Status logdetFunctor_(LaunchContext *context, NDArray *input, NDArray *output) {
  NDArray::prepareSpecialUse({output}, {input});
  auto stream = context->getCudaStream();
  // the Cholesky factors go to a C-order array of their own: the input stays as it is
  NDArray *factors = input->dup('c');
  cholesky(context, input, factors, false);

  const LongType n = input->sizeAt(-1);
  const LongType batchNum = output->lengthOf();
  dim3 launchDims = getLaunchDims("logAbsDeterminant");
  logDetKernel<T><<<launchDims.x, launchDims.y, 0, *stream>>>(reinterpret_cast<const T *>(factors->specialBuffer()), n,
                                                             batchNum, reinterpret_cast<T *>(output->specialBuffer()),
                                                             output->specialShapeInfo());
  sd::DebugHelper::checkErrorCode(stream, "logDetKernel failed");

  NDArray::registerSpecialUse({output}, {input});
  delete factors;
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
