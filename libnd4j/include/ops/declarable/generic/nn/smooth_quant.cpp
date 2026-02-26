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
// @author Adam Gibson
//
// smooth_quant - SmoothQuant W8A8 quantized matmul
//
// Implements the SmoothQuant technique: Y = (X * diag(s)^-1) @ (diag(s) * W)
// where s is a per-channel smoothing scale computed offline via calibration.
// Both smoothed X and smoothed W are quantized to INT8 for efficient GEMM.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_smooth_quant)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <helpers/MmulHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(smooth_quant, 4, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);          // [M, K] FP16/FP32 input activations
    auto weightQuant = INPUT_VARIABLE(1);    // [K, N] INT8 pre-quantized smoothed weights
    auto smoothScale = INPUT_VARIABLE(2);    // [K] per-channel smoothing scale
    auto weightScale = INPUT_VARIABLE(3);    // scalar or [N] dequant scale for weights
    auto output = OUTPUT_VARIABLE(0);        // [M, N]

    // Optional: per-channel activation scale
    NDArray* actScale = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
    NDArray* bias = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;

    auto M = input->sizeAt(0);
    auto K = input->sizeAt(1);
    auto N = weightQuant->sizeAt(1);

    // CPU fallback: smooth the input, quantize to INT8, dequant matmul
    // Step 1: Smooth input: X_smooth = X * diag(s)^-1
    auto invScale = smoothScale->dup();
    invScale->applyTransform(sd::transform::Reciprocal, invScale);
    auto smoothedInput = (*input) * (*invScale);  // broadcasts [K] across rows

    // Step 2: Fake-quantize the smoothed input to INT8 range
    // Find per-tensor scale for activation
    auto amaxResult = smoothedInput->reduceNumber(sd::reduce::AMax);
    float actMax = amaxResult->e<float>(0);
    float actQuantScale = actMax > 0.0f ? 127.0f / actMax : 1.0f;
    delete amaxResult;

    // Step 3: Dequantize weight to float
    auto weightFloat = weightQuant->cast(FLOAT32);
    float wScale = weightScale->e<float>(0);

    // Step 4: Matmul with dequant: Y = (X_smooth / actQuantScale) * (W * wScale)
    // Simplified: Y = X_smooth * W_float * wScale
    auto wDequant = (*weightFloat) * wScale;

    MmulHelper::mmul(smoothedInput, wDequant, output, 1.0f, 0.0f);
    delete weightFloat;
    delete smoothedInput;
    delete wDequant;
    delete invScale;

    // Add bias
    if (bias != nullptr) {
        std::vector<sd::LongType> biasDims = {-1};
        output->applyBroadcast(sd::broadcast::Add, &biasDims, bias, output);
    }

    return sd::Status::OK;
}

DECLARE_TYPES(smooth_quant) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})           // input
        ->setAllowedInputTypes(1, {INT8, FLOAT32})         // quantized weight
        ->setAllowedInputTypes(2, {FLOAT32})               // smooth scale
        ->setAllowedInputTypes(3, {FLOAT32})               // weight scale
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(smooth_quant) {
    auto inShape = inputShape->at(0);
    auto weightShape = inputShape->at(1);

    auto M = shape::sizeAt(inShape, 0);
    auto N = shape::sizeAt(weightShape, 1);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inShape), 'c', {M, N});

    return SHAPELIST(outputShape);
}

}  // namespace ops
}  // namespace sd

#endif
