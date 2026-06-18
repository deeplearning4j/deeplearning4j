/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @author GS <sgazeos@gmail.com>
//
#include <array/NDArray.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <system/op_boilerplate.h>

#include <ops/declarable/helpers/triangular_solve.h>
#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {


/*
 * CUDA kernel for lower triangular solve: L * x = b
 *
 * Each thread handles one (batch, rhs_column) pair.
 * Rows are processed sequentially (forward substitution has serial dependency).
 *
 * x_r = (b_r - sum_{c=0}^{r-1} L_{r,c} * x_c) / L_{r,r}
 */
template <typename T>
static SD_KERNEL void lowerTriangularSolveKernel(T const* leftBuf, LongType const* leftTads, LongType const* leftOffsets,
                                                  T const* rightBuf, LongType const* rightTads, LongType const* rightOffsets,
                                                  T* outputBuf, LongType const* outputTads, LongType const* outputOffsets,
                                                  LongType batchSize, LongType rows, LongType cols, bool unitsOnDiag) {
  // Each thread: one (batch, column) pair
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto totalThreads = batchSize * cols;

  for (auto idx = tid; idx < totalThreads; idx += gridDim.x * blockDim.x) {
    auto batch = idx / cols;
    auto j = idx % cols;

    auto leftPart = leftBuf + leftOffsets[batch];
    auto rightPart = rightBuf + rightOffsets[batch];
    auto outputPart = outputBuf + outputOffsets[batch];

    auto leftStrides = shape::stride(leftTads);
    auto rightStrides = shape::stride(rightTads);
    auto outputStrides = shape::stride(outputTads);

    // Forward substitution: solve row by row
    for (LongType r = 0; r < rows; r++) {
      // b[r, j]
      LongType rightCoords[] = {r, j};
      LongType rightIdx;
      COORDS2INDEX(2, rightStrides, rightCoords, rightIdx);
      T sum = rightPart[rightIdx];

      // Subtract known terms: sum -= L[r,c] * x[c,j] for c < r
      for (LongType c = 0; c < r; c++) {
        LongType leftCoords[] = {r, c};
        LongType leftIdx;
        COORDS2INDEX(2, leftStrides, leftCoords, leftIdx);

        LongType outCoords[] = {c, j};
        LongType outIdx;
        COORDS2INDEX(2, outputStrides, outCoords, outIdx);

        sum -= leftPart[leftIdx] * outputPart[outIdx];
      }

      // x[r, j] = sum / L[r,r]  (or just sum if unitsOnDiag)
      LongType outCoords[] = {r, j};
      LongType outIdx;
      COORDS2INDEX(2, outputStrides, outCoords, outIdx);

      if (unitsOnDiag) {
        outputPart[outIdx] = sum;
      } else {
        LongType diagCoords[] = {r, r};
        LongType diagIdx;
        COORDS2INDEX(2, leftStrides, diagCoords, diagIdx);
        outputPart[outIdx] = sum / leftPart[diagIdx];
      }
    }
  }
}


/*
 * CUDA kernel for upper triangular solve: U * x = b
 *
 * Each thread handles one (batch, rhs_column) pair.
 * Rows are processed in reverse (back substitution).
 *
 * x_r = (b_r - sum_{c=r+1}^{N-1} U_{r,c} * x_c) / U_{r,r}
 */
template <typename T>
static SD_KERNEL void upperTriangularSolveKernel(T const* leftBuf, LongType const* leftTads, LongType const* leftOffsets,
                                                  T const* rightBuf, LongType const* rightTads, LongType const* rightOffsets,
                                                  T* outputBuf, LongType const* outputTads, LongType const* outputOffsets,
                                                  LongType batchSize, LongType rows, LongType cols, bool unitsOnDiag) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto totalThreads = batchSize * cols;

  for (auto idx = tid; idx < totalThreads; idx += gridDim.x * blockDim.x) {
    auto batch = idx / cols;
    auto j = idx % cols;

    auto leftPart = leftBuf + leftOffsets[batch];
    auto rightPart = rightBuf + rightOffsets[batch];
    auto outputPart = outputBuf + outputOffsets[batch];

    auto leftStrides = shape::stride(leftTads);
    auto rightStrides = shape::stride(rightTads);
    auto outputStrides = shape::stride(outputTads);

    // Back substitution: solve row by row in reverse
    for (LongType r = rows; r > 0; r--) {
      LongType row = r - 1;

      // b[row, j]
      LongType rightCoords[] = {row, j};
      LongType rightIdx;
      COORDS2INDEX(2, rightStrides, rightCoords, rightIdx);
      T sum = rightPart[rightIdx];

      // Subtract known terms: sum -= U[row,c] * x[c,j] for c > row
      for (LongType c = row + 1; c < rows; c++) {
        LongType leftCoords[] = {row, c};
        LongType leftIdx;
        COORDS2INDEX(2, leftStrides, leftCoords, leftIdx);

        LongType outCoords[] = {c, j};
        LongType outIdx;
        COORDS2INDEX(2, outputStrides, outCoords, outIdx);

        sum -= leftPart[leftIdx] * outputPart[outIdx];
      }

      // x[row, j] = sum / U[row,row]  (or just sum if unitsOnDiag)
      LongType outCoords[] = {row, j};
      LongType outIdx;
      COORDS2INDEX(2, outputStrides, outCoords, outIdx);

      if (unitsOnDiag) {
        outputPart[outIdx] = sum;
      } else {
        LongType diagCoords[] = {row, row};
        LongType diagIdx;
        COORDS2INDEX(2, leftStrides, diagCoords, diagIdx);
        outputPart[outIdx] = sum / leftPart[diagIdx];
      }
    }
  }
}


template <typename T>
static void lowerTriangularSolve(LaunchContext* context, NDArray* leftInput, NDArray* rightInput,
                                 bool const unitsOnDiag, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {leftInput, rightInput});

  auto rows = leftInput->sizeAt(-2);
  auto cols = rightInput->sizeAt(-1);
  auto stream = context->getCudaStream();

  // For 2D (unbatched): wrap as single-batch using the array's own shape info
  if (leftInput->rankOf() == 2) {
    // Use the 2D shape info directly as "TAD" shape info
    // batchSize=1, offset=0
    auto leftBuf = reinterpret_cast<T const*>(leftInput->specialBuffer());
    auto rightBuf = reinterpret_cast<T const*>(rightInput->specialBuffer());
    auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());

    // For unbatched case, allocate single offset = 0 on device via pool
    int deviceId = sd::AffinityManager::currentDeviceId();
    LongType zeroOffset = 0;
    LongType* dOffset = reinterpret_cast<LongType*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(LongType), deviceId, *stream));
    cudaMemcpyAsync(dOffset, &zeroOffset, sizeof(LongType), cudaMemcpyHostToDevice, *stream);

    dim3 launchDims = getLaunchDims("triangular_solve");
    LongType totalWork = 1 * cols;
    dim3 grid(math::sd_min<LongType>((totalWork + launchDims.x - 1) / launchDims.x, launchDims.y));

    lowerTriangularSolveKernel<T><<<grid, launchDims.x, launchDims.z, *stream>>>(
        leftBuf, leftInput->specialShapeInfo(), dOffset,
        rightBuf, rightInput->specialShapeInfo(), dOffset,
        outputBuf, output->specialShapeInfo(), dOffset,
        1, rows, cols, unitsOnDiag);
    sd::DebugHelper::checkErrorCode(stream, "lowerTriangularSolveKernel failed");

    sd::memory::CudaMemoryPool::getInstance().free(dOffset, deviceId, *stream);
  } else {
    // Batched case: use TADs
    std::vector<LongType> dims = {-2, -1};
    auto leftTads = ConstantTadHelper::getInstance().tadForDimensions(leftInput->shapeInfo(), &dims);
    auto rightTads = ConstantTadHelper::getInstance().tadForDimensions(rightInput->shapeInfo(), &dims);
    auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dims);

    auto leftBuf = reinterpret_cast<T const*>(leftInput->specialBuffer());
    auto rightBuf = reinterpret_cast<T const*>(rightInput->specialBuffer());
    auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());

    auto batchSize = leftTads->numberOfTads();

    dim3 launchDims = getLaunchDims("triangular_solve");
    LongType totalWork = batchSize * cols;
    dim3 grid(math::sd_min<LongType>((totalWork + launchDims.x - 1) / launchDims.x, launchDims.y));

    lowerTriangularSolveKernel<T><<<grid, launchDims.x, launchDims.z, *stream>>>(
        leftBuf, leftTads->specialShapeInfo(), leftTads->specialOffsets(),
        rightBuf, rightTads->specialShapeInfo(), rightTads->specialOffsets(),
        outputBuf, outputTads->specialShapeInfo(), outputTads->specialOffsets(),
        batchSize, rows, cols, unitsOnDiag);
    sd::DebugHelper::checkErrorCode(stream, "lowerTriangularSolveKernel failed");
  }

  NDArray::registerSpecialUse({output}, {leftInput, rightInput});
}


template <typename T>
static void upperTriangularSolve(LaunchContext* context, NDArray* leftInput, NDArray* rightInput,
                                 bool const unitsOnDiag, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {leftInput, rightInput});

  auto rows = leftInput->sizeAt(-2);
  auto cols = rightInput->sizeAt(-1);
  auto stream = context->getCudaStream();

  if (leftInput->rankOf() == 2) {
    auto leftBuf = reinterpret_cast<T const*>(leftInput->specialBuffer());
    auto rightBuf = reinterpret_cast<T const*>(rightInput->specialBuffer());
    auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());

    int deviceId = sd::AffinityManager::currentDeviceId();
    LongType zeroOffset = 0;
    LongType* dOffset = reinterpret_cast<LongType*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(LongType), deviceId, *stream));
    cudaMemcpyAsync(dOffset, &zeroOffset, sizeof(LongType), cudaMemcpyHostToDevice, *stream);

    dim3 launchDims = getLaunchDims("triangular_solve");
    LongType totalWork = 1 * cols;
    dim3 grid(math::sd_min<LongType>((totalWork + launchDims.x - 1) / launchDims.x, launchDims.y));

    upperTriangularSolveKernel<T><<<grid, launchDims.x, launchDims.z, *stream>>>(
        leftBuf, leftInput->specialShapeInfo(), dOffset,
        rightBuf, rightInput->specialShapeInfo(), dOffset,
        outputBuf, output->specialShapeInfo(), dOffset,
        1, rows, cols, unitsOnDiag);
    sd::DebugHelper::checkErrorCode(stream, "upperTriangularSolveKernel failed");

    sd::memory::CudaMemoryPool::getInstance().free(dOffset, deviceId, *stream);
  } else {
    std::vector<LongType> dims = {-2, -1};
    auto leftTads = ConstantTadHelper::getInstance().tadForDimensions(leftInput->shapeInfo(), &dims);
    auto rightTads = ConstantTadHelper::getInstance().tadForDimensions(rightInput->shapeInfo(), &dims);
    auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dims);

    auto leftBuf = reinterpret_cast<T const*>(leftInput->specialBuffer());
    auto rightBuf = reinterpret_cast<T const*>(rightInput->specialBuffer());
    auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());

    auto batchSize = leftTads->numberOfTads();

    dim3 launchDims = getLaunchDims("triangular_solve");
    LongType totalWork = batchSize * cols;
    dim3 grid(math::sd_min<LongType>((totalWork + launchDims.x - 1) / launchDims.x, launchDims.y));

    upperTriangularSolveKernel<T><<<grid, launchDims.x, launchDims.z, *stream>>>(
        leftBuf, leftTads->specialShapeInfo(), leftTads->specialOffsets(),
        rightBuf, rightTads->specialShapeInfo(), rightTads->specialOffsets(),
        outputBuf, outputTads->specialShapeInfo(), outputTads->specialOffsets(),
        batchSize, rows, cols, unitsOnDiag);
    sd::DebugHelper::checkErrorCode(stream, "upperTriangularSolveKernel failed");
  }

  NDArray::registerSpecialUse({output}, {leftInput, rightInput});
}



template <typename T>
static Status triangularSolveFunctor_(LaunchContext* context, NDArray* leftInput, NDArray* rightInput,
                                          bool lower, bool unitsOnDiag, NDArray* output) {
  if (lower) {
    lowerTriangularSolve<T>(context, leftInput, rightInput, unitsOnDiag, output);
  } else {
    upperTriangularSolve<T>(context, leftInput, rightInput, unitsOnDiag, output);
  }
  return Status::OK;
}

///  triangularSolve2D - 2D implementation of triangularSolveFunctor
/// \tparam T - type of NDArray output
/// \param context - launch context pointer
/// \param leftInput  - T matrix of equation Tx = b
/// \param rightInput  - b vector of equation Tx = b
/// \param lower - lower or upper triangular matrix
/// \param unitsOnDiag - solve for case when only units (1.0) on diagonal is assumed
/// \param output - output vector (x on equation Tx = b)
///
template <typename T>
void triangularSolve2D(LaunchContext* context, NDArray& leftInput, NDArray& rightInput,
                       bool const lower, bool const unitsOnDiag, NDArray& output) {
  triangularSolveFunctor_<T>(context, const_cast<NDArray*>(&leftInput), const_cast<NDArray*>(&rightInput), lower,
                             unitsOnDiag, &output);
}

BUILD_SINGLE_TEMPLATE( void triangularSolve2D,
                      (LaunchContext* context, NDArray& leftInput, NDArray& rightInput,
                          bool const lower, bool const unitsOnDiag, NDArray& output),
                      SD_FLOAT_TYPES);

Status triangularSolveFunctor(LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool lower,
                                  bool unitsOnDiag, NDArray* output) {
  BUILD_SINGLE_SELECTOR(leftInput->dataType(), return triangularSolveFunctor_,
                        (context, leftInput, rightInput, lower, unitsOnDiag, output), SD_FLOAT_NATIVE);
}

template <typename T>
static SD_KERNEL void upperAdjointKernel(T const* input, T* output, LongType batchSize, LongType rows, LongType columns,
                                         LongType const* inputTads, LongType const* inputOffsets,
                                         LongType const* outputTads, LongType const* outputOffsets) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    auto inputPart = input + inputOffsets[b];
    auto outputPart = output + outputOffsets[b];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      for (auto c = threadIdx.y; c < columns; c += blockDim.y) {
        LongType zPos[] = {r, c};
        LongType zIndex;
        COORDS2INDEX(2, shape::stride(outputTads), zPos, zIndex);
        if (c <= r) {
          // Lower triangle + diagonal: transpose from input's upper triangle
          LongType xPos[] = {c, r};
          LongType xIndex;
          COORDS2INDEX(2, shape::stride(inputTads), xPos, xIndex);
          outputPart[zIndex] = inputPart[xIndex];
        } else {
          // Above diagonal: zero
          outputPart[zIndex] = static_cast<T>(0);
        }
      }
    }
  }
}

template <typename T>
static SD_KERNEL void lowerAdjointKernel(T const* input, T* output, LongType batchSize, LongType rows, LongType columns,
                                         LongType const* inputTads, LongType const* inputOffsets,
                                         LongType const* outputTads, LongType const* outputOffsets) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    auto inputPart = input + inputOffsets[b];
    auto outputPart = output + outputOffsets[b];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      for (auto c = threadIdx.y; c < columns; c += blockDim.y) {
        LongType zPos[] = {r, c};
        LongType zIndex;
        COORDS2INDEX(2, shape::stride(outputTads), zPos, zIndex);
        if (c >= r) {
          // Upper triangle + diagonal: transpose from input's lower triangle
          LongType xPos[] = {c, r};
          LongType xIndex;
          COORDS2INDEX(2, shape::stride(inputTads), xPos, xIndex);
          outputPart[zIndex] = inputPart[xIndex];
        } else {
          // Below diagonal: zero
          outputPart[zIndex] = static_cast<T>(0);
        }
      }
    }
  }
}

template <typename T>
static void adjointTriangularMatrix_(LaunchContext* context, NDArray * input, bool const lower, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input});
  auto stream = context->getCudaStream();
  auto inputBuf = reinterpret_cast<T const*>(input->specialBuffer());
  auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  auto rows = input->sizeAt(-2);
  auto columns = input->sizeAt(-1);
  dim3 launchDims = getLaunchDims("triangular_solve");

  if (input->rankOf() == 2) {
    // Unbatched: use the array's own shapeInfo directly (TAD for all dims on rank-2 gives wrong strides)
    int deviceId = sd::AffinityManager::currentDeviceId();
    LongType zeroOffset = 0;
    LongType* dOffset = reinterpret_cast<LongType*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(LongType), deviceId, *stream));
    cudaMemcpyAsync(dOffset, &zeroOffset, sizeof(LongType), cudaMemcpyHostToDevice, *stream);

    if (lower) {
      lowerAdjointKernel<T><<<launchDims.y, launchDims.y, launchDims.z, *stream>>>(inputBuf, outputBuf, 1, rows, columns,
                                                                                   input->specialShapeInfo(), dOffset,
                                                                                   output->specialShapeInfo(), dOffset);
      sd::DebugHelper::checkErrorCode(stream, "lowerAdjointKernel failed");
    } else {
      upperAdjointKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(inputBuf, outputBuf, 1, rows, columns,
                                                                                   input->specialShapeInfo(), dOffset,
                                                                                   output->specialShapeInfo(), dOffset);
      sd::DebugHelper::checkErrorCode(stream, "upperAdjointKernel failed");
    }
    sd::memory::CudaMemoryPool::getInstance().free(dOffset, deviceId, *stream);
  } else {
    // Batched: use TADs
    std::vector<LongType> dims = {-2, -1};
    auto inputTads = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &dims);
    auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &dims);

    if (lower) {
      lowerAdjointKernel<T><<<launchDims.y, launchDims.y, launchDims.z, *stream>>>(inputBuf, outputBuf, outputTads->numberOfTads(), rows, columns,
                                                                                   inputTads->specialShapeInfo(), inputTads->specialOffsets(),
                                                                                   outputTads->specialShapeInfo(), outputTads->specialOffsets());
      sd::DebugHelper::checkErrorCode(stream, "lowerAdjointKernel failed");
    } else {
      upperAdjointKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(inputBuf, outputBuf, outputTads->numberOfTads(), rows, columns,
                                                                                   inputTads->specialShapeInfo(), inputTads->specialOffsets(),
                                                                                   outputTads->specialShapeInfo(), outputTads->specialOffsets());
      sd::DebugHelper::checkErrorCode(stream, "upperAdjointKernel failed");
    }
  }

  NDArray::registerSpecialUse({output}, {input});
}

void adjointMatrix(LaunchContext* context, NDArray * input, bool const lower, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointTriangularMatrix_, (context, input, lower, output), SD_FLOAT_NATIVE);
}


}  // namespace helpers
}  // namespace ops
}  // namespace sd
