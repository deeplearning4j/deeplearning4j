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
  // TODO: note: this is the cpu implementation.
  // it's not preferred but cuda has enough edge cases
  // that I would prefer to have a working solution for now.
  NDArray::preparePrimaryUse({output}, {leftInput, rightInput});

  // stage 1: LU decomposition batched
  auto leftOutput = leftInput->ulike();

  auto permuShape = rightInput->getShapeAsVector();
  permuShape.pop_back();
  auto permutations = NDArrayFactory::create<LongType>('c', permuShape, context);
  lu(context, leftInput, &leftOutput, &permutations);
  auto leftLower = leftOutput.dup();

  auto rightOutput = rightInput->ulike();

  const std::vector<LongType> dims1 = {-2, -1};

  auto P = leftInput->ulike();
  P.nullify();
  auto PPart = P.allTensorsAlongDimension({-2, -1});
  auto permutationsPart = permutations.allTensorsAlongDimension({-1});
  for (auto batch = 0; batch < permutationsPart.size(); batch++) {
    for (LongType row = 0; row < PPart[batch]->rows(); row++) {
      std::vector<LongType> vec = {row, permutationsPart[batch]->t<LongType>(row)};
      PPart[batch]->r<T>(row, permutationsPart[batch]->t<LongType>(row)) = T(1.f);
    }
  }

  P.tickWriteHost();

  auto rightPart = rightInput->ulike();

  MmulHelper::matmul(&P, rightInput, &rightPart,false,false, 0.0, 0.0,&rightPart);
  ResultSet leftLowerPart = leftLower.allTensorsAlongDimension({-2, -1});
  for (auto i = 0; i < leftLowerPart.size(); i++) {
    for (LongType r = 0; r < leftLowerPart[i]->rows(); r++) leftLowerPart[i]->r<T>(r, r) = (T)1.f;
  }
  triangularSolveFunctor(context, &leftLower, &rightPart, true, false, &rightOutput);
  triangularSolveFunctor(context, &leftOutput, &rightOutput, false, false, output);
  NDArray::registerPrimaryUse({output}, {leftInput, rightInput});

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
        auto zIndex = shape::getOffset(outputTads, zPos);
        auto xIndex = shape::getOffset(outputTads, xPos);
        math::sd_swap(outputPart[zIndex], outputPart[xIndex]);
      }
    }
  }
}

template <typename T>
static void adjointMatrix_(LaunchContext* context, NDArray * input, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input});
  const std::vector<LongType> dims1 = {-2, -1};
  auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), const_cast<LongType*>(dims1.data()), dims1.size());
  auto stream = context->getCudaStream();
  auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  auto rows = input->sizeAt(-2);
  auto columns = input->sizeAt(-1);
  output->assign(input);
  dim3 solveDims = getLaunchDims("solve");

  adjointKernel<T><<<solveDims.x,solveDims.y, solveDims.z, *stream>>>(outputBuf, outputTads->numberOfTads(), rows, columns,
                                                                      outputTads->specialShapeInfo(), outputTads->specialOffsets());

  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "adjointKernel failed");

  NDArray::registerSpecialUse({output}, {input});
}

void adjointMatrix(LaunchContext* context, NDArray * input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointMatrix_, (context, input, output), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
