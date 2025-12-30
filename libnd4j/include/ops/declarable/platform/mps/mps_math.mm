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
// Metal Performance Shaders - Trigonometric and mathematical operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"
#include <cmath>

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Sine
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = sinf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(sin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SIN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Cosine
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(cos, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = cosf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(cos, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS COS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Tangent
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(tan, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = tanf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(tan, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS TAN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Arc Sine
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(asin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = asinf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(asin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ASIN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Arc Cosine
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(acos, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = acosf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(acos, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ACOS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Arc Tangent
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(atan, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = atanf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(atan, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ATAN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Arc Tangent 2 (atan2)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(atan2, ENGINE_CPU) {
    auto y = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    const float* yPtr = y->bufferAsT<float>();
    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = atan2f(yPtr[i], xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(atan2, ENGINE_CPU) {
    auto y = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);

    Requirements req("MPS ATAN2 OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Hyperbolic Sine
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sinh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = sinhf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(sinh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SINH OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Hyperbolic Cosine
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(cosh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = coshf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(cosh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS COSH OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Inverse Hyperbolic Sine
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(asinh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = asinhf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(asinh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ASINH OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Inverse Hyperbolic Cosine
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(acosh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = acoshf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(acosh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ACOSH OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Inverse Hyperbolic Tangent
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(atanh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = atanhf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(atanh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ATANH OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Floor
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(floor, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = floorf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(floor, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS FLOOR OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Ceiling
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(ceil, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = ceilf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(ceil, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS CEIL OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Round
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(round, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = roundf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(round, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ROUND OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Sign
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sign, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        if (val > 0) zPtr[i] = 1.0f;
        else if (val < 0) zPtr[i] = -1.0f;
        else zPtr[i] = 0.0f;
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(sign, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SIGN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Clip (clamp)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(clip_by_value, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    float minVal = T_ARG(0);
    float maxVal = T_ARG(1);

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        if (val < minVal) zPtr[i] = minVal;
        else if (val > maxVal) zPtr[i] = maxVal;
        else zPtr[i] = val;
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(clip_by_value, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS CLIP_BY_VALUE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reciprocal (1/x)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reciprocal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = 1.0f / xPtr[i];
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(reciprocal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS RECIPROCAL OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Square (x^2)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(square, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = xPtr[i] * xPtr[i];
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(square, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SQUARE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Cube (x^3)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(cube, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = xPtr[i] * xPtr[i] * xPtr[i];
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(cube, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS CUBE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Rsqrt (1/sqrt(x))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(rsqrt, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = 1.0f / sqrtf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(rsqrt, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS RSQRT OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Log1p (log(1 + x))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(log1p, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = log1pf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(log1p, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS LOG1P OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Expm1 (exp(x) - 1)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(expm1, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = expm1f(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(expm1, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS EXPM1 OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Erf (error function)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(erf, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = erff(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(erf, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ERF OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Erfc (complementary error function)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(erfc, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = erfcf(xPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(erfc, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ERFC OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
