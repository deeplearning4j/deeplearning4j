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
// MLIR-accelerated normalization operations
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
// batchnorm MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(batchnorm, ENGINE_CPU)

PLATFORM_IMPL(batchnorm, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [bS, iH, iW, iC] or [bS, iC, iH, iW]
    auto* mean = INPUT_VARIABLE(1);    // [iC]
    auto* variance = INPUT_VARIABLE(2); // [iC]
    auto* gamma = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr; // [iC]
    auto* beta = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;  // [iC]
    auto* output = OUTPUT_VARIABLE(0);

    double epsilon = block.numT() > 0 ? T_ARG(0) : 1e-5;
    int isNCHW = block.numI() > 0 ? INT_ARG(0) : 1;

    std::vector<NDArray*> inputs = {input, mean, variance};
    if (gamma != nullptr) inputs.push_back(gamma);
    if (beta != nullptr) inputs.push_back(beta);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("batchnorm", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(batchnorm, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR BATCHNORM");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// batchnorm_bp (backprop) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(batchnorm_bp, ENGINE_CPU)

PLATFORM_IMPL(batchnorm_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* mean = INPUT_VARIABLE(1);
    auto* variance = INPUT_VARIABLE(2);
    auto* gamma = INPUT_VARIABLE(3);
    auto* gradO = INPUT_VARIABLE(4);  // gradient of loss w.r.t. output

    auto* gradI = OUTPUT_VARIABLE(0);  // gradient w.r.t. input
    auto* gradG = OUTPUT_VARIABLE(1);  // gradient w.r.t. gamma
    auto* gradB = OUTPUT_VARIABLE(2);  // gradient w.r.t. beta

    std::vector<NDArray*> inputs = {input, mean, variance, gamma, gradO};
    std::vector<NDArray*> outputs = {gradI, gradG, gradB};

    auto status = executeMlir("batchnorm_bp", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(batchnorm_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR BATCHNORM_BP");

    // Backward ops fall through to native C++ — no MLIR backward support
    req.expectTrue(false, "Backward ops use native C++ implementation");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// layer_norm MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(layer_norm, ENGINE_CPU)

PLATFORM_IMPL(layer_norm, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* gain = INPUT_VARIABLE(1);    // gamma/scale
    auto* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr; // beta/offset
    auto* output = OUTPUT_VARIABLE(0);

    double epsilon = block.numT() > 0 ? T_ARG(0) : 1e-5;
    // normalizedShape typically inferred from last N dimensions

    std::vector<NDArray*> inputs = {input, gain};
    if (bias != nullptr) inputs.push_back(bias);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("layer_norm", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(layer_norm, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR LAYER_NORM");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// layer_norm_bp (backprop) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(layer_norm_bp, ENGINE_CPU)

PLATFORM_IMPL(layer_norm_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* gain = INPUT_VARIABLE(1);
    auto* bias = INPUT_VARIABLE(2);
    auto* gradO = INPUT_VARIABLE(3);

    auto* gradI = OUTPUT_VARIABLE(0);
    auto* gradG = OUTPUT_VARIABLE(1);
    auto* gradB = OUTPUT_VARIABLE(2);

    std::vector<NDArray*> inputs = {input, gain, bias, gradO};
    std::vector<NDArray*> outputs = {gradI, gradG, gradB};

    auto status = executeMlir("layer_norm_bp", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(layer_norm_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR LAYER_NORM_BP");

    // Backward ops fall through to native C++ — no MLIR backward support
    req.expectTrue(false, "Backward ops use native C++ implementation");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
