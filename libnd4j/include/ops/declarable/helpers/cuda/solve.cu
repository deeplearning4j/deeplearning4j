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

namespace sd {
namespace ops {
namespace helpers {




template <typename T>
static sd::Status solveFunctor_(sd::LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool adjoint,
                                NDArray* output) {

  //TODO: note: this is the cpu implementation.
  //it's not preferred but cuda has enough edge cases
  //that I would prefer to have a working solution for now.
  NDArray::preparePrimaryUse({output}, {leftInput, rightInput});

  // stage 1: LU decomposition batched
  auto leftOutput = leftInput->ulike();

  auto permuShape = rightInput->getShapeAsVector();
  permuShape.pop_back();
  auto permutations = NDArrayFactory::create<sd::LongType>('c', permuShape, context);
  helpers::lu(context, leftInput, &leftOutput, &permutations);
  auto leftLower = leftOutput.dup();

  auto rightOutput = rightInput->ulike();

  const std::vector<sd::LongType> dims1 = {-2, -1};

  auto P = leftInput->ulike();
  P.nullify();
  auto PPart = P.allTensorsAlongDimension({-2, -1});
  auto permutationsPart = permutations.allTensorsAlongDimension({-1});
  for (auto batch = 0; batch < permutationsPart.size(); batch++) {
    for (sd::LongType row = 0; row < PPart[batch]->rows(); row++) {
      std::vector<sd::LongType> vec = {row,permutationsPart[batch]->t<sd::LongType>(row)};
      PPart[batch]->r<T>(row, permutationsPart[batch]->t<sd::LongType>(row)) = T(1.f);
    }
  }



  P.tickWriteHost();

  auto rightPart = rightInput->ulike();

  MmulHelper::matmul(&P, rightInput, &rightPart, 0.0, 0);
  ResultSet leftLowerPart = leftLower.allTensorsAlongDimension({-2, -1});
  for (auto i = 0; i < leftLowerPart.size(); i++) {
    for (sd::LongType r = 0; r < leftLowerPart[i]->rows(); r++) leftLowerPart[i]->r<T>(r, r) = (T)1.f;
  }
  helpers::triangularSolveFunctor(context, &leftLower, &rightPart, true, false, &rightOutput);
  helpers::triangularSolveFunctor(context, &leftOutput, &rightOutput, false, false, output);
  NDArray::registerPrimaryUse({output}, {leftInput, rightInput});

  return sd::Status::OK;
}


sd::Status solveFunctor(sd::LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool adjoint,
                        NDArray* output) {
  BUILD_SINGLE_SELECTOR(leftInput->dataType(), return solveFunctor_, (context, leftInput, rightInput, adjoint, output),
                        SD_FLOAT_TYPES);
}

template <typename T>
static SD_KERNEL void adjointKernel(T* output, sd::LongType batchSize, sd::LongType rows, sd::LongType columns,
                                    sd::LongType const* outputTads, sd::LongType const* outputOffsets) {
  for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
    auto outputPart = output + outputOffsets[b];
    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
      for (auto c = threadIdx.y; c < r; c += blockDim.y) {
        sd::LongType zPos[] = {r, c};
        sd::LongType xPos[] = {c, r};
        auto zIndex = shape::getOffset(outputTads, zPos);
        auto xIndex = shape::getOffset(outputTads, xPos);
        math::sd_swap(outputPart[zIndex], outputPart[xIndex]);
      }
    }
  }
}

template <typename T>
static void adjointMatrix_(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input});
  const std::vector<sd::LongType> dims1 = {-2, -1};
  auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), const_cast<sd::LongType *>(dims1.data()), dims1.size());
  auto stream = context->getCudaStream();
  auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
  auto rows = input->sizeAt(-2);
  auto columns = input->sizeAt(-1);
  output->assign(input);
  dim3 solveDims = getLaunchDims("solve");

  adjointKernel<T><<<solveDims.x,solveDims.y, solveDims.z, *stream>>>(outputBuf, outputTads->numberOfTads(), rows, columns,
                                                                      outputTads->specialShapeInfo(), outputTads->specialOffsets());
  NDArray::registerSpecialUse({output}, {input});
}

void adjointMatrix(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointMatrix_, (context, input, output), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
