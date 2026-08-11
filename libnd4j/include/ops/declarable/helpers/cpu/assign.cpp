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

template <typename X>
static void assignToFloat8E4M3_(NDArray* source, NDArray* target) {
    assignImpl_<X, float8>(source, target);
}

template <typename X>
static void assignToFloat8E5M2_(NDArray* source, NDArray* target) {
    assignImpl_<X, float8_e5m2>(source, target);
}

template <typename Z>
static void assignFromFloat8E4M3_(NDArray* source, NDArray* target) {
    assignImpl_<float8, Z>(source, target);
}

template <typename Z>
static void assignFromFloat8E5M2_(NDArray* source, NDArray* target) {
    assignImpl_<float8_e5m2, Z>(source, target);
}

static bool assignFp8_(DataType xType, DataType zType, NDArray* source, NDArray* target) {
    if (zType == DataType::FLOAT8) {
        if (xType == DataType::FLOAT8) {
            assignImpl_<float8, float8>(source, target);
        } else if (xType == DataType::FLOAT8_E5M2) {
            assignImpl_<float8_e5m2, float8>(source, target);
        } else {
            BUILD_SINGLE_SELECTOR(xType, assignToFloat8E4M3_, (source, target), SD_COMMON_TYPES);
        }
        return true;
    }

    if (zType == DataType::FLOAT8_E5M2) {
        if (xType == DataType::FLOAT8) {
            assignImpl_<float8, float8_e5m2>(source, target);
        } else if (xType == DataType::FLOAT8_E5M2) {
            assignImpl_<float8_e5m2, float8_e5m2>(source, target);
        } else {
            BUILD_SINGLE_SELECTOR(xType, assignToFloat8E5M2_, (source, target), SD_COMMON_TYPES);
        }
        return true;
    }

    if (xType == DataType::FLOAT8) {
        BUILD_SINGLE_SELECTOR(zType, assignFromFloat8E4M3_, (source, target), SD_COMMON_TYPES);
        return true;
    }

    if (xType == DataType::FLOAT8_E5M2) {
        BUILD_SINGLE_SELECTOR(zType, assignFromFloat8E5M2_, (source, target), SD_COMMON_TYPES);
        return true;
    }

    return false;
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

    // FP8 conversion is deliberately scoped to cast/assign. Keeping FP8 out of
    // SD_COMMON_TYPES avoids instantiating every operation for both encodings.
    if (!assignFp8_(xType, zType, source, target)) {
        BUILD_DOUBLE_SELECTOR(xType, zType, assignImpl_, (source, target), SD_COMMON_TYPES, SD_COMMON_TYPES);
    }

    NDArray::registerSpecialUse({target}, {source});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
