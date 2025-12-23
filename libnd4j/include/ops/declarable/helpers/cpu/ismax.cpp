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
// CPU implementation of ismax helper
//

#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/helpers/ismax.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void ismax_(LaunchContext* context, NDArray* input, NDArray* output,
                   const std::vector<LongType>& dimensions) {
    // Initialize output to zeros
    output->nullify();

    if (dimensions.size() == 0 || (dimensions.size() == 1 && dimensions[0] == sd::DataTypeUtils::max<int>())) {
        // Scalar case - find the single maximum in the entire array
        auto indexMax = input->applyIndexReduce(indexreduce::IndexMax, &dimensions);
        auto targetIdx = indexMax->e<LongType>(0);

        // Set the maximum position to 1
        output->p(targetIdx, static_cast<T>(1));

        delete indexMax;
    } else {
        // Dimensional case - find maximum along specified dimensions
        std::vector<LongType> copy(dimensions);

        // Get the indices of maximum values along the specified dimensions
        auto indexMaxArr = input->applyIndexReduce(indexreduce::IndexMax, &dimensions);

        // Get TAD information for the output
        auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), copy.data(), copy.size());
        auto zTadShapeInfo = packZ->primaryShapeInfo();
        auto zTadOffsets = packZ->primaryOffsets();

        auto numTads = packZ->numberOfTads();
        auto tadLen = shape::length(zTadShapeInfo);

        auto zBuffer = output->bufferAsT<T>();

        // For each TAD, set the maximum index position to 1
        auto func = PRAGMA_THREADS_FOR {
            for (auto t = start; t < stop; t++) {
                auto zTadOffset = zTadOffsets[t];
                auto maxIdx = indexMaxArr->e<LongType>(t);

                // Calculate the actual offset within this TAD
                if (maxIdx >= 0 && maxIdx < tadLen) {
                    sd::LongType coords[SD_MAX_RANK];
                    sd::LongType zOffset;

                    const int tadRank = shape::rank(zTadShapeInfo);
                    const sd::LongType* tadShape = shape::shapeOf(zTadShapeInfo);
                    const sd::LongType* tadStride = shape::stride(zTadShapeInfo);

                    INDEX2COORDS(maxIdx, tadRank, tadShape, coords);
                    COORDS2INDEX(tadRank, tadStride, coords, zOffset);

                    zBuffer[zTadOffset + zOffset] = static_cast<T>(1);
                }
            }
        };

        samediff::Threads::parallel_for(func, 0, numTads);

        delete indexMaxArr;
    }
}

void ismax(LaunchContext* context, NDArray* input, NDArray* output, const std::vector<LongType>& dimensions) {
    NDArray::prepareSpecialUse({output}, {input});

    BUILD_SINGLE_SELECTOR(input->dataType(), ismax_, (context, input, output, dimensions), SD_COMMON_TYPES);

    NDArray::registerSpecialUse({output}, {input});
}

BUILD_SINGLE_TEMPLATE(void ismax_,
                      (sd::LaunchContext* context, NDArray* input, NDArray* output,
                       const std::vector<sd::LongType>& dimensions),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
