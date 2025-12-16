
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
// Created by GS <sgazeos@gmail.com> on 3/21/2018.
//
#include <array/ResultSet.h>
#include <exceptions/cuda_exception.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/matrix_diag_part.h>

#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// put diagonals from input batched matrices to output batched vectors
template <typename T>
static SD_KERNEL void matrixDiagPartKernel(void* inputBuffer, void* outputBuffer, LongType numTads,
                                           LongType inputLength,  LongType* tadOnlyInputShapeInfo,
                                            LongType* tadInputOffsets,
                                            LongType* tadOnlyOutputShapeInfo,
                                            LongType* tadOutputOffsets) {

  if(blockIdx.x >= numTads)
    return;
  auto outputBuffer2 = reinterpret_cast<T*>(outputBuffer);
  auto inputBuffer2 = reinterpret_cast<T const*>(inputBuffer);

  int totalThreads = blockDim.x;
  for (LongType i = blockIdx.x; i < numTads; i += gridDim.x) {
    auto yOffset = tadInputOffsets[i];
    auto xOffset = tadOutputOffsets[i];
    for (LongType j = threadIdx.x; j < inputLength; j += totalThreads) {
      LongType coords[2] = {j, j};
      LongType tadOffset, indexOffset;
      COORDS2INDEX(shape::rank(tadOnlyInputShapeInfo), shape::stride(tadOnlyInputShapeInfo), coords, tadOffset);
      COORDS2INDEX(shape::rank(tadOnlyOutputShapeInfo), shape::stride(tadOnlyOutputShapeInfo), coords, indexOffset);
      *(reinterpret_cast<T*>(outputBuffer) + xOffset + indexOffset) =
          *(reinterpret_cast<T const*>(inputBuffer) + yOffset + tadOffset);
    }
  }
}


//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page:
// https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
//
template <typename T>
static Status _matrixDiagPart(LaunchContext* context, NDArray* input, NDArray* output) {
  auto stream = context->getCudaStream();
  auto listOut = output->allTensorsAlongDimension({output->rankOf() - 1});
  auto listDiag = input->allTensorsAlongDimension({input->rankOf() - 2, input->rankOf() - 1});

  if (listOut.size() != listDiag.size()) {
    sd_printf("matrix_diag_part: Input matrix has wrong shape.", "");
    return Status::VALIDATION;
  }
  LongType lastDimension = math::sd_min(input->sizeAt(-2), input->sizeAt(-1));

  LongType dims = output->rankOf() - 1;
  std::vector<LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(output->rankOf(), 1,&dims);
  const LongType numTads =
      ShapeUtils::getNumOfSubArrs(input->shapeInfo(),*dimsToExclude);
  std::vector<LongType> outputDims({output->rankOf() - 1});
  std::vector<LongType> inputDims({input->rankOf() - 2, input->rankOf() - 1});
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &inputDims);
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &outputDims);

  if (!output->isActualOnDeviceSide()) input->syncToDevice();

  if (!input->isActualOnDeviceSide()) input->syncToDevice();

  dim3 launchDims = getLaunchDims("matrixDiag");
  matrixDiagPartKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      input->specialBuffer(),
      output->specialBuffer(),numTads, lastDimension, const_cast<sd::LongType *>(packX->specialShapeInfo()),
      const_cast<sd::LongType *>(packX->specialOffsets()),
      const_cast<sd::LongType *>(packZ->specialShapeInfo()), const_cast<sd::LongType *>(packZ->specialOffsets()));

  sd::DebugHelper::checkErrorCode(stream, "matrixDiagPartKernel failed");

  delete dimsToExclude;

  return Status::OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// caller for _matrixDiagPart
//
Status matrixDiagPart(LaunchContext* context, NDArray* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return _matrixDiagPart, (context, input, output), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( sd::Status _matrixDiagPart,
                      (sd::LaunchContext * context, NDArray* input, NDArray* output), SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
