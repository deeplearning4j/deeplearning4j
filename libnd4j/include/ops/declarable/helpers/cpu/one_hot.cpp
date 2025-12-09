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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 30.05.2019
//
// CPU implementation of one_hot helper
//

#include <execution/Threads.h>
#include <ops/declarable/helpers/one_hot.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename Z>
static void onehotImpl_(NDArray* indices, NDArray* output,
                        const LongType axis, const LongType depth,
                        const double on, const double off) {
    auto xBuffer = indices->bufferAsT<X>();
    auto zBuffer = output->bufferAsT<Z>();

    auto xShapeInfo = indices->shapeInfo();
    auto zShapeInfo = output->shapeInfo();

    const int xRank = shape::rank(xShapeInfo);
    const int zRank = shape::rank(zShapeInfo);
    const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
    const sd::LongType* zShape = shape::shapeOf(zShapeInfo);
    const sd::LongType* xStride = shape::stride(xShapeInfo);
    const sd::LongType* zStride = shape::stride(zShapeInfo);
    const sd::LongType zLen = output->lengthOf();

    const Z onVal = static_cast<Z>(on);
    const Z offVal = static_cast<Z>(off);

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            sd::LongType coord[SD_MAX_RANK];

            // Compute output coordinate and offset
            INDEX2COORDS(i, zRank, zShape, coord);
            sd::LongType zOffset;
            COORDS2INDEX(zRank, zStride, coord, zOffset);

            // Extract depth coordinate and shift axis
            const auto depthCoord = coord[axis];
            for (int j = axis; j < zRank - 1; ++j) {
                coord[j] = coord[j + 1];
            }

            // Compute input offset
            sd::LongType xOffset;
            COORDS2INDEX(xRank, xStride, coord, xOffset);

            // Check if the depth matches the index
            const LongType idx = static_cast<LongType>(xBuffer[xOffset]);
            zBuffer[zOffset] = (depthCoord == idx) ? onVal : offVal;
        }
    };

    samediff::Threads::parallel_for(func, 0, zLen);
}

void onehot(const LaunchContext* context, NDArray* indices, NDArray* output,
            const LongType axis, const LongType depth, const double on, const double off) {
    const auto xType = indices->dataType();
    const auto zType = output->dataType();

    NDArray::prepareSpecialUse({output}, {indices});

    BUILD_DOUBLE_SELECTOR(xType, zType, onehotImpl_, (indices, output, axis, depth, on, off),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);

    NDArray::registerSpecialUse({output}, {indices});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
