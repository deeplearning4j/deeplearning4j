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
// @author Adam Gibson
//
// Metal Performance Shaders - Comparison and logical operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Greater Than
//////////////////////////////////////////////////////////////////////////

static void greaterMPS(const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        bool* zPtr = z->bufferAsT<bool>();
        LongType length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] > yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] > scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = scalar > yPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(greater, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    greaterMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(greater, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS GREATER OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 input is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Greater Than or Equal
//////////////////////////////////////////////////////////////////////////

static void greaterEqualMPS(const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        bool* zPtr = z->bufferAsT<bool>();
        LongType length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] >= yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] >= scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = scalar >= yPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(greater_equal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    greaterEqualMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(greater_equal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS GREATER_EQUAL OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 input is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Less Than
//////////////////////////////////////////////////////////////////////////

static void lessMPS(const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        bool* zPtr = z->bufferAsT<bool>();
        LongType length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] < yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] < scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = scalar < yPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(less, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    lessMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(less, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS LESS OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 input is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Less Than or Equal
//////////////////////////////////////////////////////////////////////////

static void lessEqualMPS(const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        bool* zPtr = z->bufferAsT<bool>();
        LongType length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] <= yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] <= scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = scalar <= yPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(less_equal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    lessEqualMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(less_equal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS LESS_EQUAL OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 input is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Equal
//////////////////////////////////////////////////////////////////////////

static void equalMPS(const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        bool* zPtr = z->bufferAsT<bool>();
        LongType length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] == yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] == scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = scalar == yPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(equals, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    equalMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(equals, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS EQUALS OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 input is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Not Equal
//////////////////////////////////////////////////////////////////////////

static void notEqualMPS(const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        bool* zPtr = z->bufferAsT<bool>();
        LongType length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] != yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] != scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = scalar != yPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(not_equals, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    notEqualMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(not_equals, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS NOT_EQUALS OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 input is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Maximum (element-wise)
//////////////////////////////////////////////////////////////////////////

static void maximumMPS(const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        LongType length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = std::max(xPtr[i], yPtr[i]);
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = std::max(xPtr[i], scalar);
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = std::max(scalar, yPtr[i]);
            }
        }
    }
}

PLATFORM_IMPL(maximum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    maximumMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(maximum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS MAXIMUM OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Minimum (element-wise)
//////////////////////////////////////////////////////////////////////////

static void minimumMPS(const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        LongType length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = std::min(xPtr[i], yPtr[i]);
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = std::min(xPtr[i], scalar);
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (LongType i = 0; i < length; i++) {
                zPtr[i] = std::min(scalar, yPtr[i]);
            }
        }
    }
}

PLATFORM_IMPL(minimum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    minimumMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(minimum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS MINIMUM OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Where (conditional select)
//////////////////////////////////////////////////////////////////////////

static void whereMPS(const NDArray* condition, const NDArray* x, const NDArray* y, NDArray* z) {
    @autoreleasepool {
        const bool* condPtr = condition->bufferAsT<bool>();
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        LongType length = z->lengthOf();

        for (LongType i = 0; i < length; i++) {
            zPtr[i] = condPtr[i] ? xPtr[i] : yPtr[i];
        }
    }
}

PLATFORM_IMPL(where, ENGINE_CPU) {
    auto condition = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);
    auto y = INPUT_VARIABLE(2);
    auto z = OUTPUT_VARIABLE(0);

    if (condition->isEmpty()) return sd::Status::OK;

    whereMPS(condition, x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(where, ENGINE_CPU) {
    auto condition = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);

    Requirements req("MPS WHERE OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported for x/y");
    req.expectTrue(mpsUtils::isContiguous(*condition), "Condition must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
