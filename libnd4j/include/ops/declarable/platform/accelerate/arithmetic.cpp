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
// Apple Accelerate framework - Element-wise arithmetic operations via vDSP
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_ACCELERATE

/**
 * Helper to check common requirements for element-wise operations
 */
static bool checkElementWiseRequirements(Requirements& req, sd::graph::Context& block,
                                          const NDArray* x, const NDArray* y, const NDArray* z) {
    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);

    req.expectTrue(x->dataType() == DataType::FLOAT32 || x->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported by vDSP");

    req.expectTrue(x->dataType() == y->dataType(),
                   "Input arrays must have the same data type");

    req.expectTrue(accelerateUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*y), "Input Y must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*z), "Output Z must be contiguous");

    req.expectFalse(x->isEmpty(), "Input X must not be empty");
    req.expectFalse(y->isEmpty(), "Input Y must not be empty");

    // Must have same shape for element-wise ops
    req.expectTrue(x->isSameShape(y), "Input arrays must have the same shape");

    return true;
}

//////////////////////////////////////////////////////////////////////////
// ADD: z = x + y
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(add, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const LongType length = x->lengthOf();

    if (x->dataType() == DataType::FLOAT32) {
        const float* xBuf = x->bufferAsT<float>();
        const float* yBuf = y->bufferAsT<float>();
        float* zBuf = z->bufferAsT<float>();

        // vDSP_vadd: z[i] = x[i] + y[i]
        vDSP_vadd(xBuf, 1, yBuf, 1, zBuf, 1, static_cast<vDSP_Length>(length));
    } else if (x->dataType() == DataType::DOUBLE) {
        const double* xBuf = x->bufferAsT<double>();
        const double* yBuf = y->bufferAsT<double>();
        double* zBuf = z->bufferAsT<double>();

        vDSP_vaddD(xBuf, 1, yBuf, 1, zBuf, 1, static_cast<vDSP_Length>(length));
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(add, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE ADD OP");
    checkElementWiseRequirements(req, block, x, y, z);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SUBTRACT: z = x - y
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(subtract, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const LongType length = x->lengthOf();

    if (x->dataType() == DataType::FLOAT32) {
        const float* xBuf = x->bufferAsT<float>();
        const float* yBuf = y->bufferAsT<float>();
        float* zBuf = z->bufferAsT<float>();

        // vDSP_vsub: z[i] = x[i] - y[i]
        // Note: vDSP_vsub computes B - A, so we pass y first then x
        vDSP_vsub(yBuf, 1, xBuf, 1, zBuf, 1, static_cast<vDSP_Length>(length));
    } else if (x->dataType() == DataType::DOUBLE) {
        const double* xBuf = x->bufferAsT<double>();
        const double* yBuf = y->bufferAsT<double>();
        double* zBuf = z->bufferAsT<double>();

        vDSP_vsubD(yBuf, 1, xBuf, 1, zBuf, 1, static_cast<vDSP_Length>(length));
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(subtract, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE SUBTRACT OP");
    checkElementWiseRequirements(req, block, x, y, z);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// MULTIPLY: z = x * y
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(multiply, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const LongType length = x->lengthOf();

    if (x->dataType() == DataType::FLOAT32) {
        const float* xBuf = x->bufferAsT<float>();
        const float* yBuf = y->bufferAsT<float>();
        float* zBuf = z->bufferAsT<float>();

        // vDSP_vmul: z[i] = x[i] * y[i]
        vDSP_vmul(xBuf, 1, yBuf, 1, zBuf, 1, static_cast<vDSP_Length>(length));
    } else if (x->dataType() == DataType::DOUBLE) {
        const double* xBuf = x->bufferAsT<double>();
        const double* yBuf = y->bufferAsT<double>();
        double* zBuf = z->bufferAsT<double>();

        vDSP_vmulD(xBuf, 1, yBuf, 1, zBuf, 1, static_cast<vDSP_Length>(length));
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(multiply, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE MULTIPLY OP");
    checkElementWiseRequirements(req, block, x, y, z);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// DIVIDE: z = x / y
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(divide, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const LongType length = x->lengthOf();

    if (x->dataType() == DataType::FLOAT32) {
        const float* xBuf = x->bufferAsT<float>();
        const float* yBuf = y->bufferAsT<float>();
        float* zBuf = z->bufferAsT<float>();

        // vDSP_vdiv: z[i] = x[i] / y[i]
        // Note: vDSP_vdiv computes B / A, so we pass y first then x
        vDSP_vdiv(yBuf, 1, xBuf, 1, zBuf, 1, static_cast<vDSP_Length>(length));
    } else if (x->dataType() == DataType::DOUBLE) {
        const double* xBuf = x->bufferAsT<double>();
        const double* yBuf = y->bufferAsT<double>();
        double* zBuf = z->bufferAsT<double>();

        vDSP_vdivD(yBuf, 1, xBuf, 1, zBuf, 1, static_cast<vDSP_Length>(length));
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(divide, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE DIVIDE OP");
    checkElementWiseRequirements(req, block, x, y, z);
    req.logTheSuccess();
    return req;
}

#endif  // HAVE_ACCELERATE

}  // namespace platforms
}  // namespace ops
}  // namespace sd
