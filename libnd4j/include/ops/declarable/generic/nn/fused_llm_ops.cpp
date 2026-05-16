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
// Generic implementations for fused LLM operations.
// These call the platform-specific helpers (CUDA or CPU).
//

#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>

#if NOT_EXCLUDED(OP_fused_gelu) || NOT_EXCLUDED(OP_fused_layer_norm) || \
    NOT_EXCLUDED(OP_fused_rope) || NOT_EXCLUDED(OP_fused_bias_dropout_residual) || \
    NOT_EXCLUDED(OP_fused_rms_norm_swiglu) || NOT_EXCLUDED(OP_fused_attention_projection)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/fused_llm_ops.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// fused_gelu - Fast GELU approximation
//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_fused_gelu)
CONFIGURABLE_OP_IMPL(fused_gelu, 1, 1, true, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    helpers::fusedGELU(input, output, block.launchContext());

    return Status::OK;
}

DECLARE_TYPES(fused_gelu) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION);
}

CONFIGURABLE_OP_IMPL(fused_gelu_bp, 2, 1, true, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gradOut = INPUT_VARIABLE(1);
    auto gradIn = OUTPUT_VARIABLE(0);

    helpers::fusedGELUBackward(input, gradOut, gradIn, block.launchContext());

    return Status::OK;
}

DECLARE_TYPES(fused_gelu_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// fused_layer_norm - Fused layer normalization with Welford's algorithm
//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_fused_layer_norm)
CUSTOM_OP_IMPL(fused_layer_norm, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gain = INPUT_VARIABLE(1);
    NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    float epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    helpers::fusedLayerNorm(input, gain, bias, output, epsilon, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_layer_norm) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(fused_layer_norm) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
}

CUSTOM_OP_IMPL(fused_layer_norm_bp, 3, 2, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gain = INPUT_VARIABLE(1);
    auto gradOut = INPUT_VARIABLE(2);
    NDArray* bias = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

    auto gradInput = OUTPUT_VARIABLE(0);
    auto gradGain = OUTPUT_VARIABLE(1);
    NDArray* gradBias = block.outputWidth() > 2 ? OUTPUT_VARIABLE(2) : nullptr;

    float epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    helpers::fusedLayerNormBackward(input, gain, gradOut, gradInput, gradGain, gradBias,
                                     epsilon, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_layer_norm_bp) {
    auto inShape = inputShape->at(0);
    auto gainShape = inputShape->at(1);

    return SHAPELIST(
        ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(gainShape)->primary());
}

DECLARE_TYPES(fused_layer_norm_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// fused_rope - Fused rotary position embedding
//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_fused_rope)
CUSTOM_OP_IMPL(fused_rope, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int ropeType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    int rotaryDims = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;

    if (block.width() >= 3) {
        // Cached path: cos and sin provided as inputs 1 and 2
        auto cosValues = INPUT_VARIABLE(1);
        auto sinValues = INPUT_VARIABLE(2);
        helpers::fusedRoPECached(input, cosValues, sinValues, output, ropeType,
                                  block.launchContext());
    } else {
        // Position-offset path: position is read from device pointer by the kernel
        // (capture-safe — no host sync needed).
        float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
        float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

        if (block.width() == 2) {
            auto secondInput = INPUT_VARIABLE(1);
            if (secondInput->rankOf() == 0 || secondInput->isScalar()) {
                // Pass position NDArray directly — kernel reads from device pointer.
                helpers::fusedRoPE(input, output, secondInput, freqBase, freqScale, ropeType,
                                    block.launchContext(), rotaryDims);
            } else {
                // RoPE cache tensor (not a position) — fall back to iArg via scalar
                LongType posVal = block.getIArguments()->size() > 1 ? static_cast<LongType>(INT_ARG(1)) : 0;
                auto posArr = NDArrayFactory::create_<LongType>(posVal, block.launchContext());
                helpers::fusedRoPE(input, output, posArr, freqBase, freqScale, ropeType,
                                    block.launchContext(), rotaryDims);
                delete posArr;
            }
        } else {
            LongType posVal = block.getIArguments()->size() > 1 ? static_cast<LongType>(INT_ARG(1)) : 0;
            auto posArr = NDArrayFactory::create_<LongType>(posVal, block.launchContext());
            helpers::fusedRoPE(input, output, posArr, freqBase, freqScale, ropeType,
                                block.launchContext(), rotaryDims);
            delete posArr;
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_rope) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(fused_rope) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
}

CUSTOM_OP_IMPL(fused_rope_bp, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gradOut = INPUT_VARIABLE(1);
    auto gradIn = OUTPUT_VARIABLE(0);

    int ropeType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int positionOffset = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int rotaryDimsBp = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

    helpers::fusedRoPEBackward(gradOut, gradIn, positionOffset, freqBase, freqScale, ropeType,
                                block.launchContext(), rotaryDimsBp);

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_rope_bp) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(fused_rope_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// fused_bias_dropout_residual - Fused bias + dropout + residual
//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_fused_bias_dropout_residual)
CUSTOM_OP_IMPL(fused_bias_dropout_residual, 3, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto bias = INPUT_VARIABLE(1);
    auto residual = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    LongType seed = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    float dropoutProb = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;
    bool training = block.numB() > 0 ? B_ARG(0) : false;

    helpers::fusedBiasDropoutResidual(input, bias, residual, output, dropoutProb, seed,
                                       training, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_bias_dropout_residual) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(fused_bias_dropout_residual) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}
#endif

//////////////////////////////////////////////////////////////////////////
// fused_rms_norm_swiglu - Fused RMS norm + SwiGLU FFN
//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_fused_rms_norm_swiglu)
CUSTOM_OP_IMPL(fused_rms_norm_swiglu, 4, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gamma = INPUT_VARIABLE(1);
    auto wGate = INPUT_VARIABLE(2);
    auto wUp = INPUT_VARIABLE(3);
    auto output = OUTPUT_VARIABLE(0);

    float epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    helpers::fusedRmsNormSwiGLU(input, gamma, wGate, wUp, output, epsilon, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_rms_norm_swiglu) {
    auto inShape = inputShape->at(0);
    auto wGateShape = inputShape->at(2);
    auto dtype = ArrayOptions::dataType(inShape);

    // Output shape: [batch, seq_len, intermediate_dim]
    auto batch = shape::sizeAt(inShape, static_cast<LongType>(0));
    auto seqLen = shape::sizeAt(inShape, static_cast<LongType>(1));
    auto intermediateDim = shape::sizeAt(wGateShape, static_cast<LongType>(1));

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {batch, seqLen, intermediateDim}));
}

DECLARE_TYPES(fused_rms_norm_swiglu) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
}

CUSTOM_OP_IMPL(fused_rms_norm_swiglu_bp, 5, 4, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gamma = INPUT_VARIABLE(1);
    auto wGate = INPUT_VARIABLE(2);
    auto wUp = INPUT_VARIABLE(3);
    auto gradOut = INPUT_VARIABLE(4);

    auto gradInput = OUTPUT_VARIABLE(0);
    auto gradGamma = OUTPUT_VARIABLE(1);
    auto gradWGate = OUTPUT_VARIABLE(2);
    auto gradWUp = OUTPUT_VARIABLE(3);

    float epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    helpers::fusedRmsNormSwiGLUBackward(input, gamma, wGate, wUp, gradOut,
                                         gradInput, gradGamma, gradWGate, gradWUp,
                                         epsilon, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_rms_norm_swiglu_bp) {
    auto inShape = inputShape->at(0);
    auto gammaShape = inputShape->at(1);
    auto wGateShape = inputShape->at(2);
    auto wUpShape = inputShape->at(3);

    return SHAPELIST(
        ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(gammaShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(wGateShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(wUpShape)->primary());
}

DECLARE_TYPES(fused_rms_norm_swiglu_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// fused_attention_projection - Attention output x O-projection matmul + bias
//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_fused_attention_projection)
CUSTOM_OP_IMPL(fused_attention_projection, 2, 1, false, 0, 0) {
    auto attentionOutput = INPUT_VARIABLE(0);
    auto Wo              = INPUT_VARIABLE(1);
    NDArray* bias        = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output          = OUTPUT_VARIABLE(0);

    helpers::fusedAttentionProjection(attentionOutput, Wo, bias, output, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_attention_projection) {
    auto attnShape = inputShape->at(0);   // [B, S, H, D]  or  [B, S, hidden]
    auto woShape   = inputShape->at(1);   // [hidden_dim, out_dim]
    auto dtype     = ArrayOptions::dataType(attnShape);

    const LongType batch  = shape::sizeAt(attnShape, static_cast<LongType>(0));
    const LongType seqLen = shape::sizeAt(attnShape, static_cast<LongType>(1));
    // Wo is always 2D: [hidden_dim, out_dim]
    const LongType outDim = shape::sizeAt(woShape, static_cast<LongType>(1));

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {batch, seqLen, outDim}));
}

DECLARE_TYPES(fused_attention_projection) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
}
#endif

}  // namespace ops
}  // namespace sd

#endif
