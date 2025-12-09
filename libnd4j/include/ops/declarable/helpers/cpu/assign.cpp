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
// CPU implementation of assign helper
//

#include <ops/declarable/helpers/assign.h>
#include <execution/Threads.h>
#include <helpers/ShapeUtils.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename Z>
static void assignImpl_(NDArray* source, NDArray* target) {
    auto xBuffer = source->bufferAsT<X>();
    auto zBuffer = target->bufferAsT<Z>();

    auto xShapeInfo = source->shapeInfo();
    auto zShapeInfo = target->shapeInfo();

    const int xRank = shape::rank(xShapeInfo);
    const int zRank = shape::rank(zShapeInfo);
    const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
    const sd::LongType* zShape = shape::shapeOf(zShapeInfo);
    const sd::LongType* xStride = shape::stride(xShapeInfo);
    const sd::LongType* zStride = shape::stride(zShapeInfo);
    const sd::LongType len = target->lengthOf();

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            sd::LongType xCoords[SD_MAX_RANK], zCoords[SD_MAX_RANK];
            sd::LongType xOffset, zOffset;

            INDEX2COORDS(i, zRank, zShape, zCoords);
            INDEX2COORDS(i, xRank, xShape, xCoords);
            COORDS2INDEX(xRank, xStride, xCoords, xOffset);
            COORDS2INDEX(zRank, zStride, zCoords, zOffset);

            zBuffer[zOffset] = static_cast<Z>(xBuffer[xOffset]);
        }
    };

    samediff::Threads::parallel_for(func, 0, len);
}

void assign(sd::LaunchContext* context, sd::NDArray* target, sd::NDArray* source) {
    if (target->lengthOf() != source->lengthOf()) {
        std::string errorMsg = "assign helper: Source and target arrays must have the same length. ";
        errorMsg += "Source shape: " + ShapeUtils::shapeAsString(source) + ", ";
        errorMsg += "Target shape: " + ShapeUtils::shapeAsString(target) + ", ";
        errorMsg += "Source datatype: " + DataTypeUtils::asString(source->dataType()) + ", ";
        errorMsg += "Target datatype: " + DataTypeUtils::asString(target->dataType());
        THROW_EXCEPTION(errorMsg.c_str());
    }

    NDArray::prepareSpecialUse({target}, {source});

    auto xType = source->dataType();
    auto zType = target->dataType();

    BUILD_DOUBLE_SELECTOR(xType, zType, assignImpl_, (source, target), SD_COMMON_TYPES, SD_COMMON_TYPES);

    NDArray::registerSpecialUse({target}, {source});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
