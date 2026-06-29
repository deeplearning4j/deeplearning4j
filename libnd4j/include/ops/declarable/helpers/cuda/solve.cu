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
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/MmulHelper.h>
#include <system/op_boilerplate.h>

#include "../lup.h"
#include "../solve.h"
#include "../triangular_solve.h"
#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {


// ------------------------------------------------------------------------------------------------------------------ //
// Kernel: write 1.0 on the diagonal of each [n x n] batch slice stored in a TAD-described layout.
// For unbatched (2D) use: batchSize=1, tads=array's own specialShapeInfo(), offsets=single zero offset.
template <typename T>
static SD_KERNEL void setDiagonalOneKernel(T* buf, LongType const* tads, LongType const* offsets,
                                           LongType batchSize, LongType n) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    const LongType base = (offsets != nullptr) ? offsets[b] : 0;
    auto slice = buf + base;
    for (auto r = threadIdx.x; r < n; r += blockDim.x) {
      LongType pos[] = {r, r};
      LongType idx;
      COORDS2INDEX(shape::rank(tads), shape::stride(tads), pos, idx);
      slice[idx] = T(1.f);
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------ //
// Kernel: build permutation matrix P on device.
// Each (batch, row) thread reads col = permBuf[permOffset + row_idx] and writes P[pOffset + idx] = T(1).
// pTadShape/pTadOffsets     — {-2,-1} TADs of P (rank-2 slices, shape [rows x rows]).
// permTadShape/permTadOffsets — {-1} TADs of permutations (rank-1 slices, shape [rows]).
// For unbatched 2D: batchSize=1, pTadOffsets=nullptr, pTadShape=P->specialShapeInfo(),
//                               permTadOffsets=nullptr, permTadShape=permutations->specialShapeInfo().
template <typename T>
static SD_KERNEL void setPermutationOnesKernel(T* pBuf,
                                               LongType const* pTadShape, LongType const* pTadOffsets,
                                               LongType const* permBuf,
                                               LongType const* permTadShape, LongType const* permTadOffsets,
                                               LongType batchSize, LongType rows) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    const LongType pBase    = (pTadOffsets    != nullptr) ? pTadOffsets[b]    : 0;
    const LongType permBase = (permTadOffsets != nullptr) ? permTadOffsets[b] : 0;
    auto pSlice    = pBuf    + pBase;
    auto permSlice = permBuf + permBase;
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      // read column index from permutation vector
      LongType permPos[] = {r};
      LongType permIdx;
      COORDS2INDEX(shape::rank(permTadShape), shape::stride(permTadShape), permPos, permIdx);
      LongType col = permSlice[permIdx];

      // write the 1 into P at (row=r, col=col)
      LongType pPos[] = {r, col};
      LongType pIdx;
      COORDS2INDEX(shape::rank(pTadShape), shape::stride(pTadShape), pPos, pIdx);
      pSlice[pIdx] = T(1.f);
    }
  }
}


template <typename T>
static Status solveFunctor_(LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool adjoint,
                            NDArray* output) {
  NDArray::preparePrimaryUse({output}, {leftInput, rightInput});

  // stage 1: LU decomposition batched
  auto leftOutput = leftInput->ulike();

  bool unbatched = (leftInput->rankOf() == 2);

  auto permuShapePtr = rightInput->getShapeAsVector();
  std::vector<LongType> permuShape(*permuShapePtr);
  delete permuShapePtr;
  permuShape.pop_back();

  // For batched inputs (rank > 2), pop_back() leaves a multi-D shape (e.g., [batch, 3]).
  // For non-batched (2D) inputs, pop_back() leaves a 1D shape (e.g., [3]) — keep it 1D.
  // The kernel setPermutationOnesKernel uses COORDS2INDEX with permPos={r} (1 element);
  // if permTadShape were rank-2 ([1,3]), COORDS2INDEX would read coords[1] (uninitialized)
  // giving r*stride[0] = r*3 instead of r*1, causing wrong permutation reads.
  // Only force 2D for batched inputs where allTensorsAlongDimension({-1}) is needed.
  if (permuShape.size() == 1 && !unbatched) {
    permuShape.insert(permuShape.begin(), 1);
  }

  auto permutations = NDArrayFactory::create<LongType>('c', permuShape, context);
  lu(context, leftInput, leftOutput, permutations);
  auto leftLower = leftOutput->dup();

  auto rightOutput = rightInput->ulike();

  const std::vector<LongType> dims1 = {-2, -1};

  auto P = leftInput->ulike();
  P->nullify();

  // Build permutation matrix P entirely on the device — no host element access.
  {
    NDArray::prepareSpecialUse({P}, {permutations});
    auto stream  = context->getCudaStream();
    auto pBuf    = reinterpret_cast<T*>(P->specialBuffer());
    auto permBuf = reinterpret_cast<LongType*>(permutations->specialBuffer());
    LongType rows = P->sizeAt(-2);
    dim3 solveDims = getLaunchDims("solve");
    if (unbatched) {
      // 2D: single batch — offsets=nullptr means kernel uses base=0.
      setPermutationOnesKernel<T><<<1, solveDims.y, 0, *stream>>>(
          pBuf,
          P->specialShapeInfo(), nullptr,
          permBuf,
          permutations->specialShapeInfo(), nullptr,
          1, rows);
      sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "setPermutationOnesKernel (unbatched) failed");
    } else {
      // Batched: get {-2,-1} TADs for P and {-1} TADs for permutations.
      const std::vector<LongType> p2d    = {-2, -1};
      const std::vector<LongType> perm1d = {-1};
      auto pTads    = ConstantTadHelper::getInstance().tadForDimensions(
          P->shapeInfo(),            const_cast<LongType*>(p2d.data()),    p2d.size());
      auto permTads = ConstantTadHelper::getInstance().tadForDimensions(
          permutations->shapeInfo(), const_cast<LongType*>(perm1d.data()), perm1d.size());
      LongType batchSize = pTads->numberOfTads();
      setPermutationOnesKernel<T><<<(int)batchSize, solveDims.y, 0, *stream>>>(
          pBuf,
          pTads->specialShapeInfo(),    pTads->specialOffsets(),
          permBuf,
          permTads->specialShapeInfo(), permTads->specialOffsets(),
          batchSize, rows);
      sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "setPermutationOnesKernel (batched) failed");
    }
    NDArray::registerSpecialUse({P}, {permutations});
  }

  auto rightPart = rightInput->ulike();

  MmulHelper::matmul(P, rightInput, rightPart, false, false, 1.0, 0.0, rightPart);

  // Set diagonal of lower triangular part to 1 directly on the device buffer.
  // leftLower was produced by dup() which copies via D2D; its host buffer is NOT populated
  // with LU data, so writing via r<T>() + syncToDevice() would overwrite the valid device
  // LU data with uninitialized host memory. Use a kernel on specialBuffer() instead.
  {
    // leftLower's device buffer holds valid LU data (from dup, D2D) and is modified in place by
    // the kernel; prepare/register handle coherence (its host buffer is intentionally untouched).
    NDArray::prepareSpecialUse({leftLower}, {});
    auto stream = context->getCudaStream();
    auto buf = reinterpret_cast<T*>(leftLower->specialBuffer());
    LongType n = leftLower->sizeAt(-2);
    dim3 solveDims = getLaunchDims("solve");

    if (unbatched) {
      // 2D: single batch — offsets=nullptr means kernel uses base=0.
      setDiagonalOneKernel<T><<<1, solveDims.y, 0, *stream>>>(
          buf, leftLower->specialShapeInfo(), nullptr, 1, n);
      sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "setDiagonalOneKernel failed");
    } else {
      const std::vector<LongType> dimdims = {-2, -1};
      auto tads = ConstantTadHelper::getInstance().tadForDimensions(
          leftLower->shapeInfo(), const_cast<LongType*>(dimdims.data()), dimdims.size());
      LongType batchSize = tads->numberOfTads();
      setDiagonalOneKernel<T><<<(int)batchSize, solveDims.y, 0, *stream>>>(
          buf, tads->specialShapeInfo(), tads->specialOffsets(), batchSize, n);
      sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "setDiagonalOneKernel failed");
    }
    NDArray::registerSpecialUse({leftLower}, {});
  }

  triangularSolveFunctor(context, leftLower, rightPart, true, false, rightOutput);
  triangularSolveFunctor(context, leftOutput, rightOutput, false, false, output);
  NDArray::registerPrimaryUse({output}, {leftInput, rightInput});

  delete leftOutput;
  delete permutations;
  delete leftLower;
  delete rightOutput;
  delete P;
  delete rightPart;

  return Status::OK;
}

Status solveFunctor(LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool adjoint,
                        NDArray* output) {
  BUILD_SINGLE_SELECTOR(leftInput->dataType(), return solveFunctor_, (context, leftInput, rightInput, adjoint, output),
                        SD_FLOAT_TYPES);
}

template <typename T>
static SD_KERNEL void adjointKernel(T* output, LongType batchSize, LongType rows, LongType columns,
                                    LongType const* outputTads, LongType const* outputOffsets) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    auto outputPart = output + outputOffsets[b];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      for (auto c = threadIdx.y; c < r; c += blockDim.y) {
        LongType zPos[] = {r, c};
        LongType xPos[] = {c, r};
        LongType zIndex, xIndex;
        COORDS2INDEX(shape::rank(outputTads), shape::stride(outputTads), zPos, zIndex);
        COORDS2INDEX(shape::rank(outputTads), shape::stride(outputTads), xPos, xIndex);
        math::sd_swap(outputPart[zIndex], outputPart[xIndex]);
      }
    }
  }
}

template <typename T>
static SD_KERNEL void adjointKernelUnbatched(T* output, LongType rows, LongType columns,
                                              LongType const* outputShape) {
  for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
    for (auto c = threadIdx.y; c < r; c += blockDim.y) {
      LongType zPos[] = {r, c};
      LongType xPos[] = {c, r};
      LongType zIndex, xIndex;
      COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zPos, zIndex);
      COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), xPos, xIndex);
      math::sd_swap(output[zIndex], output[xIndex]);
    }
  }
}

template <typename T>
static void adjointMatrix_(LaunchContext* context, NDArray * input, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input});
  auto stream = context->getCudaStream();
  auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  auto rows = input->sizeAt(-2);
  auto columns = input->sizeAt(-1);
  output->assign(input);
  dim3 solveDims = getLaunchDims("solve");

  // For unbatched (2D) inputs, allTensorsAlongDimension({-2,-1}) produces rank-0 TADs
  // which breaks coordinate-based indexing. Use the array shape directly instead.
  if (input->rankOf() == 2) {
    adjointKernelUnbatched<T><<<1, solveDims.y, solveDims.z, *stream>>>(
        outputBuf, rows, columns, output->specialShapeInfo());
  } else {
    const std::vector<LongType> dims1 = {-2, -1};
    auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(
        output->shapeInfo(), const_cast<LongType*>(dims1.data()), dims1.size());

    adjointKernel<T><<<solveDims.x, solveDims.y, solveDims.z, *stream>>>(
        outputBuf, outputTads->numberOfTads(), rows, columns,
        outputTads->specialShapeInfo(), outputTads->specialOffsets());
  }

  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "adjointKernel failed");

  NDArray::registerSpecialUse({output}, {input});
}

void adjointMatrix(LaunchContext* context, NDArray * input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointMatrix_, (context, input, output), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
