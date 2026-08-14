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
// Metal Performance Shaders - Extended activation functions
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
// Hard Sigmoid: max(0, min(1, x * 0.2 + 0.5))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(hard_sigmoid, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i] * 0.2f + 0.5f;
        zPtr[i] = std::max(0.0f, std::min(1.0f, val));
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(hard_sigmoid, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS HARD_SIGMOID OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Hard Swish: x * hard_sigmoid(x) = x * max(0, min(1, x/6 + 0.5))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(hardswish, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        float hs = std::max(0.0f, std::min(1.0f, val / 6.0f + 0.5f));
        zPtr[i] = val * hs;
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(hardswish, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS HARDSWISH OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(mish, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        float softplus = log1pf(expf(val));
        zPtr[i] = val * tanhf(softplus);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(mish, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS MISH OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Softplus: log(1 + exp(x))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(softplus, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        zPtr[i] = log1pf(expf(xPtr[i]));
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(softplus, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SOFTPLUS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Softsign: x / (1 + |x|)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(softsign, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        zPtr[i] = val / (1.0f + fabsf(val));
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(softsign, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SOFTSIGN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// PReLU: max(0, x) + alpha * min(0, x)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(prelu, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto alpha = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    const float* alphaPtr = alpha->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();
    LongType alphaLen = alpha->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        float a = alphaPtr[i % alphaLen];  // Broadcasting
        zPtr[i] = val >= 0 ? val : a * val;
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(prelu, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS PRELU OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(selu, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        if (val >= 0) {
            zPtr[i] = scale * val;
        } else {
            zPtr[i] = scale * alpha * (expf(val) - 1.0f);
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(selu, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SELU OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// CELU: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(celu, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    float alpha = 1.0f;
    if (block.getTArguments()->size() > 0) {
        alpha = T_ARG(0);
    }

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        if (val >= 0) {
            zPtr[i] = val;
        } else {
            zPtr[i] = alpha * (expf(val / alpha) - 1.0f);
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(celu, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS CELU OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ReLU6: min(max(0, x), 6)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(relu6, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        zPtr[i] = std::min(std::max(0.0f, val), 6.0f);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(relu6, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS RELU6 OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Thresholded ReLU: x if x > theta, else 0
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(thresholdedrelu, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    float theta = 1.0f;
    if (block.getTArguments()->size() > 0) {
        theta = T_ARG(0);
    }

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();
    LongType length = z->lengthOf();

    for (LongType i = 0; i < length; i++) {
        float val = xPtr[i];
        zPtr[i] = val > theta ? val : 0.0f;
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(thresholdedrelu, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS THRESHOLDEDRELU OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Log Softmax
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(log_softmax, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    const float* xPtr = x->bufferAsT<float>();
    float* zPtr = z->bufferAsT<float>();

    // Softmax along last axis
    LongType outerSize = 1;
    LongType innerSize = x->sizeAt(-1);

    for (int i = 0; i < x->rankOf() - 1; i++) {
        outerSize *= x->sizeAt(i);
    }

    for (LongType outer = 0; outer < outerSize; outer++) {
        LongType offset = outer * innerSize;

        // Find max for numerical stability
        float maxVal = xPtr[offset];
        for (LongType i = 1; i < innerSize; i++) {
            if (xPtr[offset + i] > maxVal) {
                maxVal = xPtr[offset + i];
            }
        }

        // Compute sum of exp(x - max)
        float sum = 0.0f;
        for (LongType i = 0; i < innerSize; i++) {
            sum += expf(xPtr[offset + i] - maxVal);
        }

        float logSum = logf(sum);

        // Compute log_softmax
        for (LongType i = 0; i < innerSize; i++) {
            zPtr[offset + i] = xPtr[offset + i] - maxVal - logSum;
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(log_softmax, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS LOG_SOFTMAX OP");
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
