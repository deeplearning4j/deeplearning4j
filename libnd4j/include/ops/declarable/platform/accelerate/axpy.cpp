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
// Apple Accelerate framework - AXPY implementation
// Uses vecLib BLAS cblas_saxpy/cblas_daxpy for y = alpha*x + y
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE

namespace sd {
namespace ops {
namespace platforms {

PLATFORM_IMPL(axpy, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    // Get alpha from T arguments (default 1.0)
    double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    const auto n = static_cast<int>(x->lengthOf());
    const int incX = 1;
    const int incY = 1;

    // If z != y, copy y to z first
    if (z != y) {
        z->assign(y);
    }

    if (x->dataType() == DataType::FLOAT32) {
        cblas_saxpy(n,
                    static_cast<float>(alpha),
                    x->bufferAsT<float>(), incX,
                    z->bufferAsT<float>(), incY);
    } else if (x->dataType() == DataType::DOUBLE) {
        cblas_daxpy(n,
                    alpha,
                    x->bufferAsT<double>(), incX,
                    z->bufferAsT<double>(), incY);
    }

    return Status::OK;
}

PLATFORM_CHECK(axpy, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements reqs("AXPY ACCELERATE");
    // Check Accelerate requirements
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);

    // Both arrays must be contiguous
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);

    // Same supported data types
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->dataType() == y->dataType(), "Inputs must have same data type");

    // Must have same shape
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
