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
// MKL VML accelerated error function (erf)
// Uses Intel MKL VML's vsErf/vdErf when available, otherwise falls back to scalar
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/platform/mkldnn/mkldnnUtils.h>
#include <system/platform_boilerplate.h>
#include <system/op_enums.h>
#include <helpers/MklVmlHelper.h>
#include <helpers/shape.h>
#include <math/templatemath.h>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
// Helper to check if array is truly contiguous in memory (no gaps, unit stride)
static bool isContiguous(const sd::LongType* shapeInfo) {
    if (shapeInfo == nullptr) return false;

    const int rank = shape::rank(shapeInfo);
    if (rank == 0) return true;  // Scalar is contiguous

    const sd::LongType* strides = shape::stride(shapeInfo);
    const sd::LongType* shapeOf = shape::shapeOf(shapeInfo);

    // Check if array is contiguous by verifying strides match expected contiguous strides
    sd::LongType expectedStride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        if (strides[i] != expectedStride) {
            return false;
        }
        expectedStride *= shapeOf[i];
    }
    return true;
}

//////////////////////////////////////////////////////////////////////
// Erf implementation using coordinate-based indexing for proper view support
template <typename T>
static void erfImpl(NDArray* x, NDArray* z) {
    const auto length = x->lengthOf();
    const auto xShapeInfo = x->shapeInfo();
    const auto zShapeInfo = z->shapeInfo();
    const int rank = shape::rank(xShapeInfo);

    const T* xBuffer = x->bufferAsT<T>();
    T* zBuffer = z->bufferAsT<T>();

    // For contiguous arrays, we can use MKL VML directly
    if (isContiguous(xShapeInfo) && isContiguous(zShapeInfo)) {
        sd::vml::erf(length, xBuffer, zBuffer);
        return;
    }

    const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
    const sd::LongType* xStrides = shape::stride(xShapeInfo);
    const sd::LongType* zStrides = shape::stride(zShapeInfo);

    // For non-contiguous arrays (views), use coordinate-based indexing
    if (rank == 1) {
        // Optimized 1D case
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (sd::LongType i = 0; i < length; ++i) {
            zBuffer[i * zStrides[0]] = sd::math::sd_erf<T, T>(xBuffer[i * xStrides[0]]);
        }
    } else {
        // General case using coords
        PRAGMA_OMP_PARALLEL_FOR
        for (sd::LongType i = 0; i < length; ++i) {
            sd::LongType coords[SD_MAX_RANK];
            sd::LongType xOffset, zOffset;

            INDEX2COORDS(i, rank, xShape, coords);
            COORDS2INDEX(rank, xStrides, coords, xOffset);
            COORDS2INDEX(rank, zStrides, coords, zOffset);

            zBuffer[zOffset] = sd::math::sd_erf<T, T>(xBuffer[xOffset]);
        }
    }
}

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(Erf, ENGINE_ONEDNN) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->dataType() == DataType::FLOAT32) {
        erfImpl<float>(input, output);
    } else if (input->dataType() == DataType::DOUBLE) {
        erfImpl<double>(input, output);
    } else {
        // Fallback for other types
        output->assign(input);
        output->applyTransform(sd::transform::Erf, output);
    }

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(Erf, ENGINE_ONEDNN) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("ONEDNN ERF OP (MKL VML)");
    
        req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
        req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
        // MKL VML only supports float32 and double
        req.expectTrue(makeInfoVariable(x->dataType() == DataType::FLOAT32 || x->dataType() == DataType::DOUBLE, TYPE_MSG_INPUT),
                       "Must be FLOAT32 or DOUBLE") &&
        req.expectTrue(makeInfoVariable(z->dataType() == DataType::FLOAT32 || z->dataType() == DataType::DOUBLE, TYPE_MSG_OUTPUT),
                       "Must be FLOAT32 or DOUBLE") &&
        req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT),
                     makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT));
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////
// Erfc (complementary error function) - 1 - erf(x)
//////////////////////////////////////////////////////////////////////
template <typename T>
static void erfcImpl(NDArray* x, NDArray* z) {
    const auto length = x->lengthOf();
    const auto xShapeInfo = x->shapeInfo();
    const auto zShapeInfo = z->shapeInfo();
    const int rank = shape::rank(xShapeInfo);

    const T* xBuffer = x->bufferAsT<T>();
    T* zBuffer = z->bufferAsT<T>();

    // For contiguous arrays, we can use MKL VML directly
    if (isContiguous(xShapeInfo) && isContiguous(zShapeInfo)) {
        // Compute erf first
        sd::vml::erf(length, xBuffer, zBuffer);

        // Then compute 1 - erf(x)
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (sd::LongType i = 0; i < length; ++i) {
            zBuffer[i] = static_cast<T>(1) - zBuffer[i];
        }
        return;
    }

    const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
    const sd::LongType* xStrides = shape::stride(xShapeInfo);
    const sd::LongType* zStrides = shape::stride(zShapeInfo);

    // For non-contiguous arrays (views), use coordinate-based indexing
    if (rank == 1) {
        // Optimized 1D case
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (sd::LongType i = 0; i < length; ++i) {
            zBuffer[i * zStrides[0]] = static_cast<T>(1) - sd::math::sd_erf<T, T>(xBuffer[i * xStrides[0]]);
        }
    } else {
        // General case using coords
        PRAGMA_OMP_PARALLEL_FOR
        for (sd::LongType i = 0; i < length; ++i) {
            sd::LongType coords[SD_MAX_RANK];
            sd::LongType xOffset, zOffset;

            INDEX2COORDS(i, rank, xShape, coords);
            COORDS2INDEX(rank, xStrides, coords, xOffset);
            COORDS2INDEX(rank, zStrides, coords, zOffset);

            zBuffer[zOffset] = static_cast<T>(1) - sd::math::sd_erf<T, T>(xBuffer[xOffset]);
        }
    }
}

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(Erfc, ENGINE_ONEDNN) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->dataType() == DataType::FLOAT32) {
        erfcImpl<float>(input, output);
    } else if (input->dataType() == DataType::DOUBLE) {
        erfcImpl<double>(input, output);
    } else {
        // Fallback for other types
        output->assign(input);
        output->applyTransform(sd::transform::Erfc, output);
    }

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(Erfc, ENGINE_ONEDNN) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("ONEDNN ERFC OP (MKL VML)");
    
        req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
        req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
        // MKL VML only supports float32 and double
        req.expectTrue(makeInfoVariable(x->dataType() == DataType::FLOAT32 || x->dataType() == DataType::DOUBLE, TYPE_MSG_INPUT),
                       "Must be FLOAT32 or DOUBLE") &&
        req.expectTrue(makeInfoVariable(z->dataType() == DataType::FLOAT32 || z->dataType() == DataType::DOUBLE, TYPE_MSG_OUTPUT),
                       "Must be FLOAT32 or DOUBLE") &&
        req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT),
                     makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT));
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
