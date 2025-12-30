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
// Apple Accelerate framework - Dot product implementation
// Uses vecLib BLAS cblas_sdot/cblas_ddot for optimized dot product
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <helpers/MmulHelper.h>

#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE

namespace sd {
namespace ops {
namespace platforms {

PLATFORM_IMPL(dot, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    // Validate inputs are vectors (1D arrays)
    REQUIRE_TRUE(x->rankOf() == 1 && y->rankOf() == 1, 0,
                 "DOT ACCELERATE: both inputs must be 1D vectors, got ranks %i and %i",
                 x->rankOf(), y->rankOf());
    REQUIRE_TRUE(x->lengthOf() == y->lengthOf(), 0,
                 "DOT ACCELERATE: vectors must have equal length, got %lld and %lld",
                 x->lengthOf(), y->lengthOf());

    const auto n = static_cast<int>(x->lengthOf());
    const int incX = 1;
    const int incY = 1;

    if (x->dataType() == DataType::FLOAT32) {
        float result = cblas_sdot(n,
                                   x->bufferAsT<float>(), incX,
                                   y->bufferAsT<float>(), incY);
        z->p(0, result);
    } else if (x->dataType() == DataType::DOUBLE) {
        double result = cblas_ddot(n,
                                    x->bufferAsT<double>(), incX,
                                    y->bufferAsT<double>(), incY);
        z->p(0, result);
    }

    return Status::OK;
}

PLATFORM_CHECK(dot, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements reqs("DOT ACCELERATE");
    // Check Accelerate requirements
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);

    // Must be 1D vectors
    reqs.expectTrue(x->rankOf() == 1, RANK_MSG_INPUT0);
    reqs.expectTrue(y->rankOf() == 1, RANK_MSG_INPUT1);

    // Both arrays must be contiguous
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);

    // Same supported data types
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->dataType() == y->dataType(), "Inputs must have same data type");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
