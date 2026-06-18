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
// Apple Accelerate framework - Rounding functions
// Uses vForce for optimized ceil, floor, round, rint, trunc
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
// CEIL - Ceiling function (round up)
//==============================================================================

PLATFORM_IMPL(ceil, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvceilf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvceil(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(ceil, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("CEIL ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// FLOOR - Floor function (round down)
//==============================================================================

PLATFORM_IMPL(floor, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvfloorf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvfloor(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(floor, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("FLOOR ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// RINT - Round to nearest integer
//==============================================================================

PLATFORM_IMPL(rint, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvnintf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvnint(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(rint, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("RINT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ROUND - Round to nearest integer (away from zero for .5)
//==============================================================================

PLATFORM_IMPL(round, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        // vvnintf rounds to nearest, which is close to round behavior
        vvnintf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvnint(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(round, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("ROUND ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// TRUNCATE - Truncate toward zero
//==============================================================================

PLATFORM_IMPL(truncate, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvintf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvint(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(truncate, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("TRUNCATE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// RECIPROCAL - 1/x
//==============================================================================

PLATFORM_IMPL(reciprocal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvrecf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvrec(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(reciprocal, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("RECIPROCAL ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// RSQRT - Reciprocal square root (1/sqrt(x))
//==============================================================================

PLATFORM_IMPL(rsqrt, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvrsqrtf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvrsqrt(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(rsqrt, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("RSQRT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SQUARE - x^2
//==============================================================================

PLATFORM_IMPL(square, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        // vDSP_vsq: z[i] = x[i] * x[i]
        vDSP_vsq(x->bufferAsT<float>(), 1, z->bufferAsT<float>(), 1, n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vDSP_vsqD(x->bufferAsT<double>(), 1, z->bufferAsT<double>(), 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(square, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("SQUARE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// LOG10 - Logarithm base 10
//==============================================================================

PLATFORM_IMPL(log10, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvlog10f(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvlog10(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(log10, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("LOG10 ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// LOG1P - log(1 + x)
//==============================================================================

PLATFORM_IMPL(log1p, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvlog1pf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvlog1p(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(log1p, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("LOG1P ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// EXPM1 - exp(x) - 1
//==============================================================================

PLATFORM_IMPL(expm1, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvexpm1f(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvexpm1(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(expm1, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("EXPM1 ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
