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
    auto status = executeMlir("xw_plus_b", block, inputs, outputs);

    if (status != Status::OK()) {
        sd_printf("MLIR xw_plus_b execution failed\n", "");
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR xw_plus_b failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(xw_plus_b, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);
    auto* bias = INPUT_VARIABLE(2);

    Requirements req("MLIR XW_PLUS_B");

    // Check MLIR availability
    req.expectTrue(mlirEnabled(), "MLIR enabled") &&

    // Check supported data types
    req.expectIn(x->dataType(),
                 {FLOAT32, DOUBLE, HALF, BFLOAT16},
                 "Input X dtype supported by MLIR") &&

    req.expectIn(weights->dataType(),
                 {FLOAT32, DOUBLE, HALF, BFLOAT16},
                 "Weights dtype supported by MLIR") &&

    // Check matching types for x and weights
    req.expectEq(x->dataType(), weights->dataType(),
                 "Matching X and Weights dtypes") &&

    // Check rank requirements
    req.expectEq(x->rankOf(), 2,
                 "X is 2D [batch, in_features]") &&

    req.expectEq(weights->rankOf(), 2,
                 "Weights is 2D [in_features, out_features]") &&

    req.expectEq(bias->rankOf(), 1,
                 "Bias is 1D [out_features]") &&

    // Check shape compatibility
    req.expectEq(x->sizeAt(1), weights->sizeAt(0),
                 "X.in_features == Weights.in_features") &&

    req.expectEq(weights->sizeAt(1), bias->sizeAt(0),
                 "Weights.out_features == Bias.size") &&

    // Check minimum size threshold
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE,
                   "Tensor size above MLIR threshold") &&

    // Check contiguous memory
    req.expectTrue(x->ews() == 1 || x->ews() == 0,
                   "Input X has contiguous memory") &&
    req.expectTrue(weights->ews() == 1 || weights->ews() == 0,
                   "Weights has contiguous memory");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
