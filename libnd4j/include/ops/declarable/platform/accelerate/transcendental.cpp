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
// Apple Accelerate framework - Transcendental functions
// Uses vForce (vectorized math) and vDSP for optimized exp, log, sqrt, abs, neg, pow
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE

namespace sd {
namespace ops {
namespace platforms {

//==============================================================================
// EXP - Exponential function
//==============================================================================

PLATFORM_IMPL(exp, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvexpf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvexp(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(exp, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("EXP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// LOG - Natural logarithm
//==============================================================================

PLATFORM_IMPL(log, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvlogf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvlog(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(log, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("LOG ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SQRT - Square root
//==============================================================================

PLATFORM_IMPL(sqrt, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvsqrtf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvsqrt(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(sqrt, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("SQRT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ABS - Absolute value
//==============================================================================

PLATFORM_IMPL(abs, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvfabsf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvfabs(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(abs, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("ABS ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// NEG - Negation
//==============================================================================

PLATFORM_IMPL(neg, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        // vDSP_vneg: z[i] = -x[i]
        vDSP_vneg(x->bufferAsT<float>(), 1, z->bufferAsT<float>(), 1, n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vDSP_vnegD(x->bufferAsT<double>(), 1, z->bufferAsT<double>(), 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(neg, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("NEG ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// POW - Power function
//==============================================================================

PLATFORM_IMPL(pow, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    // Handle broadcast case - if y is scalar, use scalar power
    if (y->isScalar()) {
        if (x->dataType() == DataType::FLOAT32) {
            float exponent = y->e<float>(0);
            // For scalar exponent, we need to create a filled array
            std::vector<float> exponents(n, exponent);
            vvpowf(z->bufferAsT<float>(), exponents.data(), x->bufferAsT<float>(), &n);
        } else if (x->dataType() == DataType::DOUBLE) {
            double exponent = y->e<double>(0);
            std::vector<double> exponents(n, exponent);
            vvpow(z->bufferAsT<double>(), exponents.data(), x->bufferAsT<double>(), &n);
        }
    } else {
        // Element-wise power
        if (x->dataType() == DataType::FLOAT32) {
            vvpowf(z->bufferAsT<float>(), y->bufferAsT<float>(), x->bufferAsT<float>(), &n);
        } else if (x->dataType() == DataType::DOUBLE) {
            vvpow(z->bufferAsT<double>(), y->bufferAsT<double>(), x->bufferAsT<double>(), &n);
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(pow, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("POW ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    // y must be either scalar or same shape as x
    reqs.expectTrue(y->isScalar() || x->isSameShape(y), SHAPE_MSG_INPUT1);

    // If not scalar, y must also be contiguous
    if (!y->isScalar()) {
        reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    }

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
