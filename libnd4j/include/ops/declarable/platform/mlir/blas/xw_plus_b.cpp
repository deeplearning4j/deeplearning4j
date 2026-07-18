/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// MLIR-accelerated fused matrix multiply + bias add (xw_plus_b)
// Common operation in neural network linear/dense layers
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <ops/declarable/platform/mlir/mlirUtils.h>

#if defined(HAVE_MLIR)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// xw_plus_b MLIR implementation
// Computes: Y = X @ W + B (fused matmul + bias add)
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(xw_plus_b, ENGINE_CPU)

PLATFORM_IMPL(xw_plus_b, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);       // Input [batch, in_features]
    auto* weights = INPUT_VARIABLE(1); // Weights [in_features, out_features]
    auto* bias = INPUT_VARIABLE(2);    // Bias [out_features]
    auto* output = OUTPUT_VARIABLE(0); // Output [batch, out_features]

    // Prepare inputs and outputs for MLIR execution
    std::vector<NDArray*> inputs = {x, weights, bias};
    std::vector<NDArray*> outputs = {output};

    // Execute via MLIR JIT
    // The MLIR dialect will generate a fused kernel that:
    // 1. Performs matmul(x, weights)
    // 2. Broadcasts and adds bias
    // All in a single kernel to minimize memory bandwidth
    auto status = executeMlirEx("xw_plus_b", block, inputs, outputs);

    if (status != Status::OK) {
        sd_printf("MLIR xw_plus_b execution failed\n", "");
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(xw_plus_b, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);
    auto* bias = INPUT_VARIABLE(2);

    Requirements req("MLIR XW_PLUS_B");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
