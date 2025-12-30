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
// MLIR-accelerated extended convolution operations
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
// conv1d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(conv1d, ENGINE_CPU)

PLATFORM_IMPL(conv1d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [bS, iW, iC] or [bS, iC, iW]
    auto* weights = INPUT_VARIABLE(1); // [kW, iC, oC]
    auto* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    int kW = INT_ARG(0);
    int sW = INT_ARG(1);
    int pW = INT_ARG(2);
    int dW = INT_ARG(3);
    int paddingMode = INT_ARG(4);  // 0-VALID, 1-SAME, 2-CAUSAL
    int isNCW = block.numI() > 5 ? INT_ARG(5) : 1;

    std::vector<NDArray*> inputs = {input, weights};
    if (bias != nullptr) inputs.push_back(bias);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("conv1d", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR conv1d failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(conv1d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);

    Requirements req("MLIR CONV1D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->rankOf() == 3, "Input rank == 3") &&
    req.expectTrue(weights->rankOf() == 3, "Weights rank == 3") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// conv1d_bp MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(conv1d_bp, ENGINE_CPU)

PLATFORM_IMPL(conv1d_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);
    auto* gradO = INPUT_VARIABLE(2);

    auto* gradI = OUTPUT_VARIABLE(0);
    auto* gradW = OUTPUT_VARIABLE(1);
    auto* gradB = block.numOutputs() > 2 ? OUTPUT_VARIABLE(2) : nullptr;

    std::vector<NDArray*> inputs = {input, weights, gradO};
    std::vector<NDArray*> outputs = {gradI, gradW};
    if (gradB != nullptr) outputs.push_back(gradB);

    auto status = executeMlir("conv1d_bp", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR conv1d_bp failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(conv1d_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR CONV1D_BP");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// conv3d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(conv3dnew, ENGINE_CPU)

PLATFORM_IMPL(conv3dnew, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [bS, iD, iH, iW, iC] or [bS, iC, iD, iH, iW]
    auto* weights = INPUT_VARIABLE(1); // [kD, kH, kW, iC, oC]
    auto* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    int kD = INT_ARG(0);
    int kH = INT_ARG(1);
    int kW = INT_ARG(2);
    int sD = INT_ARG(3);
    int sH = INT_ARG(4);
    int sW = INT_ARG(5);
    int pD = INT_ARG(6);
    int pH = INT_ARG(7);
    int pW = INT_ARG(8);
    int dD = INT_ARG(9);
    int dH = INT_ARG(10);
    int dW = INT_ARG(11);
    int paddingMode = INT_ARG(12);
    int isNCDHW = block.numI() > 13 ? INT_ARG(13) : 1;

    std::vector<NDArray*> inputs = {input, weights};
    if (bias != nullptr) inputs.push_back(bias);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("conv3d", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR conv3d failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(conv3dnew, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);

    Requirements req("MLIR CONV3D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->rankOf() == 5, "Input rank == 5") &&
    req.expectTrue(weights->rankOf() == 5, "Weights rank == 5") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// conv3d_bp MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(conv3dnew_bp, ENGINE_CPU)

PLATFORM_IMPL(conv3dnew_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);
    auto* gradO = INPUT_VARIABLE(2);

    auto* gradI = OUTPUT_VARIABLE(0);
    auto* gradW = OUTPUT_VARIABLE(1);
    auto* gradB = block.numOutputs() > 2 ? OUTPUT_VARIABLE(2) : nullptr;

    std::vector<NDArray*> inputs = {input, weights, gradO};
    std::vector<NDArray*> outputs = {gradI, gradW};
    if (gradB != nullptr) outputs.push_back(gradB);

    auto status = executeMlir("conv3d_bp", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR conv3d_bp failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(conv3dnew_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR CONV3D_BP");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// depthwise_conv2d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CPU)

PLATFORM_IMPL(depthwise_conv2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [bS, iH, iW, iC] or [bS, iC, iH, iW]
    auto* weights = INPUT_VARIABLE(1); // [kH, kW, iC, mC] - depthMultiplier
    auto* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    int kH = INT_ARG(0);
    int kW = INT_ARG(1);
    int sH = INT_ARG(2);
    int sW = INT_ARG(3);
    int pH = INT_ARG(4);
    int pW = INT_ARG(5);
    int dH = INT_ARG(6);
    int dW = INT_ARG(7);
    int paddingMode = INT_ARG(8);
    int isNCHW = block.numI() > 9 ? INT_ARG(9) : 1;

    std::vector<NDArray*> inputs = {input, weights};
    if (bias != nullptr) inputs.push_back(bias);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("depthwise_conv2d", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR depthwise_conv2d failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(depthwise_conv2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);

    Requirements req("MLIR DEPTHWISE_CONV2D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->rankOf() == 4, "Input rank == 4") &&
    req.expectTrue(weights->rankOf() == 4, "Weights rank == 4") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// depthwise_conv2d_bp MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_CPU)

PLATFORM_IMPL(depthwise_conv2d_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);
    auto* gradO = INPUT_VARIABLE(2);

    auto* gradI = OUTPUT_VARIABLE(0);
    auto* gradW = OUTPUT_VARIABLE(1);
    auto* gradB = block.numOutputs() > 2 ? OUTPUT_VARIABLE(2) : nullptr;

    std::vector<NDArray*> inputs = {input, weights, gradO};
    std::vector<NDArray*> outputs = {gradI, gradW};
    if (gradB != nullptr) outputs.push_back(gradB);

    auto status = executeMlir("depthwise_conv2d_bp", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR depthwise_conv2d_bp failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(depthwise_conv2d_bp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR DEPTHWISE_CONV2D_BP");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// separable_conv2d MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(sconv2d, ENGINE_CPU)

PLATFORM_IMPL(sconv2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* depthwiseWeights = INPUT_VARIABLE(1);
    auto* pointwiseWeights = INPUT_VARIABLE(2);
    auto* bias = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input, depthwiseWeights, pointwiseWeights};
    if (bias != nullptr) inputs.push_back(bias);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("separable_conv2d", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR separable_conv2d failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(sconv2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SEPARABLE_CONV2D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->rankOf() == 4, "Input rank == 4") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// deconv2d (transposed conv) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(deconv2d, ENGINE_CPU)

PLATFORM_IMPL(deconv2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);
    auto* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input, weights};
    if (bias != nullptr) inputs.push_back(bias);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("deconv2d", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR deconv2d failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(deconv2d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR DECONV2D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->rankOf() == 4, "Input rank == 4") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// deconv3d (transposed conv 3d) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(deconv3d, ENGINE_CPU)

PLATFORM_IMPL(deconv3d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* weights = INPUT_VARIABLE(1);
    auto* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input, weights};
    if (bias != nullptr) inputs.push_back(bias);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("deconv3d", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR deconv3d failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(deconv3d, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR DECONV3D");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->rankOf() == 5, "Input rank == 5") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
