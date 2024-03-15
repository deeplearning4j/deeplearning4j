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
// @author Yurii Shyrma, created on 21.09.2018
// @author raver119@gmail.com
//

#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/TAD.h>
#include <loops/special_kernels.h>
#include <ops/declarable/helpers/ismax.h>

#include <execution/cuda/LaunchDims.h>


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void ismax_(LaunchContext* context, const NDArray* input, NDArray* output,
                   const std::vector<LongType>& dimensions) {
  auto stream = context->getCudaStream();

  auto xRank = input->rankOf();
  auto zRank = output->rankOf();
  auto xType = input->dataType();
  auto zType = output->dataType();
  input->syncToDevice();
  LongType* special = nullptr;
  PointersManager manager(context, "IsMaxHelper");
  if (dimensions.size() == 0) {
    /**
     * In case of vector-input for IsMax, it just turns into IndexReduce call + subsequent filler call
     */
    auto indexMax = input->applyIndexReduce(indexreduce::IndexMax, &dimensions);
    auto targetIdx = indexMax.e<LongType>(0);

    dim3 launchDims = getLaunchDims("ismaxFill");
    BUILD_SINGLE_SELECTOR(
        zType, fillIsMaxGeneric,
        (launchDims, stream, output->specialBuffer(), output->specialShapeInfo(), output->lengthOf(), targetIdx),
        SD_COMMON_TYPES);
    manager.synchronize();

  } else {
    LongType* hostYShapeInfo = nullptr;
    LongType* hostTShapeInfo = nullptr;
    LongType* dimension = nullptr;

    LongType dimensionLength = dimensions.size();
    std::vector<LongType> copy(dimensions);

    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), copy.data(), copy.size());

    // we launch legacy IndexMax op, to get indices of max values along dimension
    auto indexMaxArr = input->applyIndexReduce(indexreduce::IndexMax, &dimensions);

    dim3 launchDims = getLaunchDims("ismax");
    dimension = (LongType*)manager.replicatePointer(dimensions.data(), dimensions.size() * sizeof(LongType));

    // at this point, all IMax indexes are gathered, and we execute filler
    BUILD_SINGLE_SELECTOR(
        zType, fillDimensionalIsMaxGeneric,
        (launchDims, stream, indexMaxArr.specialBuffer(), output->specialBuffer(), output->specialShapeInfo(),
         packZ->specialShapeInfo(), dimension, dimensionLength, packZ->specialOffsets()),
        SD_COMMON_TYPES);
    manager.synchronize();
  }
}

void ismax(LaunchContext* context, const NDArray* input, NDArray* output, const std::vector<LongType>& dimensions) {
  NDArray::prepareSpecialUse({output}, {input});

  BUILD_SINGLE_SELECTOR(input->dataType(), ismax_, (context, input, output, dimensions), SD_COMMON_TYPES);

  NDArray::registerSpecialUse({output}, {input});
}

BUILD_SINGLE_TEMPLATE(template void ismax_,
                      (sd::LaunchContext * context, const NDArray* input, NDArray* output,
                       const std::vector<sd::LongType>& dimensions),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
