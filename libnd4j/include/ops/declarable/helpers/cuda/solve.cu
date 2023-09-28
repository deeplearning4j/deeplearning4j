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
static SD_KERNEL void oneOnDiagonalKernel(T* ioBuf, sd::LongType const* ioShape, sd::LongType const* tadShape,
                                          sd::LongType const* tadOffsets, sd::LongType batchNum, sd::LongType rowNum) {
  for (auto i = blockIdx.x; i < batchNum; i += gridDim.x) {
    auto matrixPart = ioBuf + tadOffsets[i];
    for (auto j = threadIdx.x; j < rowNum; j += blockDim.x) {
      sd::LongType pos[] = {j, j};
      auto offset = shape::getOffset(tadShape, pos);

      matrixPart[offset] = T(1.f);
    }
  }
}

template <typename T>
static SD_KERNEL void restorePermutationsKernel(T* PBuf, sd::LongType const* PShapeInfo,
                                                const LongType* permutationsBuf,
                                                sd::LongType const* PTadShapeInfo, sd::LongType const* PTadSOffsets,
                                                sd::LongType const* permutationsTadShapeInfo,
                                                sd::LongType const* permutationsTadOffsets, sd::LongType batchNum,
                                                sd::LongType rowNum) {
  for (auto batch = blockIdx.x; batch < batchNum; batch += gridDim.x) {
    auto permutations = permutationsBuf + permutationsTadOffsets[batch];
    auto P = PBuf + PTadSOffsets[batch];

    for (auto row = threadIdx.x; row < rowNum; row += blockDim.x) {
      sd::LongType posZ[] = {row, permutations[row]};
      auto zOffset = shape::getOffset(PTadShapeInfo, posZ);
      P[zOffset] = T(1.f);
    }
  }
}

template <typename T>
static sd::Status solveFunctor_(sd::LaunchContext* context, NDArray* leftInput, NDArray* rightInput, bool adjoint,
                                NDArray* output) {
  // stage 1: LU decomposition batched
  auto leftOutput = leftInput->ulike();
  auto permuShape = rightInput->getShapeAsVector();
  permuShape.pop_back();
  auto permutations = NDArrayFactory::create<sd::LongType>('c', permuShape, context);
  helpers::lu(context, leftInput, &leftOutput, &permutations);
  auto P = leftInput->ulike();  // permutations batched matrix
  P.nullify();                  // to fill up matrices with zeros
  auto PPart = P.allTensorsAlongDimension({-2, -1});
  auto permutationsPart = permutations.allTensorsAlongDimension({-1});

  for (auto batch = 0; batch < permutationsPart.size(); ++batch) {
    for (sd::LongType row = 0; row < PPart[batch]->rows(); ++row) {
      PPart[batch]->r<T>(row, permutationsPart[batch]->t<sd::LongType>(row)) = T(1.f);
    }
  }

  auto leftLower = leftOutput.dup();
  auto rightOutput = rightInput->ulike();
  auto rightPermuted = rightOutput.ulike();
  MmulHelper::matmul(&P, rightInput, &rightPermuted, 0, 0);
  ResultSet leftLowerPart = leftLower.allTensorsAlongDimension({-2, -1});
  for (auto i = 0; i < leftLowerPart.size(); i++) {
    for (sd::LongType r = 0; r < leftLowerPart[i]->rows(); r++) leftLowerPart[i]->r<T>(r, r) = (T)1.f;
  }
  // stage 2: triangularSolveFunctor for Lower with given b
  helpers::triangularSolveFunctor(context, &leftLower, &rightPermuted, true, false, &rightOutput);
  // stage 3: triangularSolveFunctor for Upper with output of previous stage
  helpers::triangularSolveFunctor(context, &leftOutput, &rightOutput, false, false, output);
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
  auto inputPart = input->allTensorsAlongDimension({-2, -1});
  auto outputPart = output->allTensorsAlongDimension({-2, -1});
  auto rows = input->sizeAt(-2);
  output->assign(input);

  auto batchLoop = PRAGMA_THREADS_FOR {
    for (auto batch = start; batch < stop; batch++) {
      for (sd::LongType r = 0; r < rows; r++) {
        for (sd::LongType c = 0; c < r; c++) {
          math::sd_swap(outputPart[batch]->r<T>(r, c), outputPart[batch]->r<T>(c, r));
        }
      }
    }
  };
  samediff::Threads::parallel_tad(batchLoop, 0, inputPart.size(), 1);
}

void adjointMatrix(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), adjointMatrix_, (context, input, output), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
