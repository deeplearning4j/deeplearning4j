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




template <typename T>
static Status solveFunctor_(LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool adjoint,
                            NDArray* output) {
  NDArray::preparePrimaryUse({output}, {leftInput, rightInput});

  // stage 1: LU decomposition batched
  auto leftOutput = leftInput->ulike();

  auto permuShapePtr = rightInput->getShapeAsVector();
  std::vector<LongType> permuShape(*permuShapePtr);
  delete permuShapePtr;
  permuShape.pop_back();

  // For non-batched (2D) inputs, pop_back() leaves a 1D shape (e.g., [3]).
  // allTensorsAlongDimension({-1}) on a 1D array doesn't properly decompose
  // into individual vectors. Ensure the shape is at least 2D by prepending a 1
  // so that the batch decomposition works correctly (e.g., [3] -> [1, 3]).
  if (permuShape.size() == 1) {
    permuShape.insert(permuShape.begin(), 1);
  }

  auto permutations = NDArrayFactory::create<LongType>('c', permuShape, context);
  lu(context, leftInput, leftOutput, permutations);
  auto leftLower = leftOutput->dup();

  auto rightOutput = rightInput->ulike();

  const std::vector<LongType> dims1 = {-2, -1};

  auto P = leftInput->ulike();
  P->nullify();

  // For unbatched (2D) inputs, allTensorsAlongDimension({-2,-1}) produces rank-0 TADs
  // which breaks coordinate-based indexing. Use the arrays directly instead.
  bool unbatched = (leftInput->rankOf() == 2);
  if (unbatched) {
    for (LongType row = 0; row < P->rows(); row++) {
      P->r<T>(row, permutations->t<LongType>(row)) = T(1.f);
    }
  } else {
    auto PPart = P->allTensorsAlongDimension({-2, -1});
    auto permutationsPart = permutations->allTensorsAlongDimension({-1});
    for (auto batch = 0; batch < permutationsPart.size(); batch++) {
      for (LongType row = 0; row < PPart[batch]->rows(); row++) {
        PPart[batch]->r<T>(row, permutationsPart[batch]->t<LongType>(row)) = T(1.f);
      }
    }
  }

  P->tickWriteHost();
  P->syncToDevice();

  auto rightPart = rightInput->ulike();

  MmulHelper::matmul(P, rightInput, rightPart, false, false, 1.0, 0.0, rightPart);

  // Set diagonal of lower triangular part to 1
  if (unbatched) {
    for (LongType r = 0; r < leftLower->rows(); r++) leftLower->r<T>(r, r) = (T)1.f;
  } else {
    ResultSet leftLowerPart = leftLower->allTensorsAlongDimension({-2, -1});
    for (auto i = 0; i < leftLowerPart.size(); i++) {
      for (LongType r = 0; r < leftLowerPart[i]->rows(); r++) leftLowerPart[i]->r<T>(r, r) = (T)1.f;
    }
  }
  leftLower->syncToDevice();
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
