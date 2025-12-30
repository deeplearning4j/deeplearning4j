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
// MLIR-accelerated 2D convolution
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
// conv2d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(conv2d, ENGINE_CPU)

PLATFORM_IMPL(conv2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);    // [bS, iH, iW, iC] or [bS, iC, iH, iW]
    auto* weights = INPUT_VARIABLE(1);  // [kH, kW, iC, oC] or [oC, iC, kH, kW]
    auto* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);  // [bS, oH, oW, oC] or [bS, oC, oH, oW]

    // Get conv parameters from integer arguments
    int kH = INT_ARG(0);  // kernel height
    int kW = INT_ARG(1);  // kernel width
    int sH = INT_ARG(2);  // stride height
    int sW = INT_ARG(3);  // stride width
    int pH = INT_ARG(4);  // padding height
    int pW = INT_ARG(5);  // padding width
    int dH = INT_ARG(6);  // dilation height
    int dW = INT_ARG(7);  // dilation width
    int paddingMode = INT_ARG(8);  // 0-VALID, 1-SAME, 2-CAUSAL
    int isNCHW = block.numI() > 9 ? INT_ARG(9) : 1;  // data format
    int wFormat = block.numI() > 10 ? INT_ARG(10) : 0;  // weights format

    // Prepare inputs and outputs for MLIR execution
    std::vector<NDArray*> inputs = {input, weights};
    if (bias != nullptr) {
        inputs.push_back(bias);
    }
    std::vector<NDArray*> outputs = {output};

    // Execute via MLIR JIT
    auto status = executeMlir("conv2d", block, inputs, outputs);

    if (status != Status::OK()) {
        sd_printf("MLIR conv2d execution failed\n", "");
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR conv2d failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(conv2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);

    Requirements req("MLIR CONV2D");

    // Check MLIR availability
    req.expectTrue(mlirEnabled(), "MLIR enabled") &&

    // Check supported data types
    req.expectIn(input->dataType(),
                 {FLOAT32, DOUBLE, HALF, BFLOAT16},
                 "Input dtype supported by MLIR") &&

    req.expectIn(weights->dataType(),
                 {FLOAT32, DOUBLE, HALF, BFLOAT16},
                 "Weights dtype supported by MLIR") &&

    // Check matching types
    req.expectEq(input->dataType(), weights->dataType(),
                 "Matching input and weights dtypes") &&

    // Check rank requirements
    req.expectEq(input->rankOf(), 4, "Input is 4D [N,C,H,W] or [N,H,W,C]") &&
    req.expectEq(weights->rankOf(), 4, "Weights is 4D") &&

    // Check minimum size threshold
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE,
                   "Tensor size above MLIR threshold") &&

    // Check contiguous memory
    req.expectTrue(input->ews() == 1 || input->ews() == 0,
                   "Input has contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// conv2d_bp (backprop) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(conv2d_bp, ENGINE_CPU)

PLATFORM_IMPL(conv2d_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);
    auto* gradO = INPUT_VARIABLE(2);  // gradient of loss w.r.t. output

    auto* gradI = OUTPUT_VARIABLE(0);  // gradient of loss w.r.t. input
    auto* gradW = OUTPUT_VARIABLE(1);  // gradient of loss w.r.t. weights
    auto* gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // gradient w.r.t. bias

    std::vector<NDArray*> inputs = {input, weights, gradO};
    std::vector<NDArray*> outputs = {gradI, gradW};
    if (gradB != nullptr) {
        outputs.push_back(gradB);
    }

    auto status = executeMlir("conv2d_bp", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR conv2d_bp failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(conv2d_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);

    Requirements req("MLIR CONV2D_BP");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectEq(input->dataType(), weights->dataType(), "Matching dtypes") &&
    req.expectEq(input->rankOf(), 4, "Input is 4D") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
