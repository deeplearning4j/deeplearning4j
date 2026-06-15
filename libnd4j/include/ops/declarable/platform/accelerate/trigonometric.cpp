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
// Apple Accelerate framework - Trigonometric functions
// Uses vForce for optimized sin, cos, tan, asin, acos, atan, sinh, cosh
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
// SIN - Sine function
//==============================================================================

PLATFORM_IMPL(sin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvsinf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvsin(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(sin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("SIN ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// COS - Cosine function
//==============================================================================

PLATFORM_IMPL(cos, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvcosf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvcos(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(cos, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("COS ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// TAN - Tangent function
//==============================================================================

PLATFORM_IMPL(tan, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvtanf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvtan(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(tan, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("TAN ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ASIN - Arc sine (inverse sine)
//==============================================================================

PLATFORM_IMPL(asin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvasinf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvasin(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(asin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("ASIN ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ACOS - Arc cosine (inverse cosine)
//==============================================================================

PLATFORM_IMPL(acos, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvacosf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvacos(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(acos, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("ACOS ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ATAN - Arc tangent (inverse tangent)
//==============================================================================

PLATFORM_IMPL(atan, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvatanf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvatan(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(atan, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("ATAN ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SINH - Hyperbolic sine
//==============================================================================

PLATFORM_IMPL(sinh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvsinhf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvsinh(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(sinh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("SINH ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// COSH - Hyperbolic cosine
//==============================================================================

PLATFORM_IMPL(cosh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvcoshf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvcosh(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(cosh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("COSH ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ATANH - Inverse hyperbolic tangent
//==============================================================================

PLATFORM_IMPL(atanh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvatanhf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvatanh(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(atanh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("ATANH ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ASINH - Inverse hyperbolic sine
//==============================================================================

PLATFORM_IMPL(asinh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvasinhf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvasinh(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(asinh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("ASINH ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ACOSH - Inverse hyperbolic cosine
//==============================================================================

PLATFORM_IMPL(acosh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vvacoshf(z->bufferAsT<float>(), x->bufferAsT<float>(), &n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vvacosh(z->bufferAsT<double>(), x->bufferAsT<double>(), &n);
    }

    return Status::OK;
}

PLATFORM_CHECK(acosh, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("ACOSH ACCELERATE");
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
