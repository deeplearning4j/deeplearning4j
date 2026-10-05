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
    NOT_EXCLUDED(OP_fused_rms_norm_swiglu) || NOT_EXCLUDED(OP_fused_attention_projection) || \
    NOT_EXCLUDED(OP_fused_mrope) || NOT_EXCLUDED(OP_vision_embedding_merge)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/fused_llm_ops.h>

namespace sd {
namespace ops {

#if NOT_EXCLUDED(OP_fused_layer_norm) || NOT_EXCLUDED(OP_fused_rope) || \
    NOT_EXCLUDED(OP_fused_bias_dropout_residual) || NOT_EXCLUDED(OP_fused_rms_norm_swiglu) || \
    NOT_EXCLUDED(OP_fused_mrope) || NOT_EXCLUDED(OP_vision_embedding_merge)
// The shape of an output that is a fresh array shaped like `source`: its type, shape and order on dense strides.
// The source's own strides, flags and view offset say where its elements sit in some other buffer, while the output
// gets a buffer of its own holding exactly length() elements: the cached shape of the source itself gave the output of
// a stepped view (every second column of a wider array) strides that address several times that, and the helpers'
// copy back into it ran past the end of the buffer.
static LongType* denseShapeLike(const LongType* source) {
    return ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(source), shape::order(source), shape::rank(source), shape::shapeOf(source),
        shape::isEmptyConst(source) ? ARRAY_EMPTY : 0);
}
#endif

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
    getOpDescriptor()->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION | OP_TRAIT_BACKWARD);
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

    // The kernels normalize rows of the last dimension's length, read a gain and a bias of that length and write one
    // output element per input element: any other size reads or writes out of bounds.
    REQUIRE_TRUE(input->rankOf() >= 1, 0, "fused_layer_norm: input must have rank >= 1, got rank %i",
                 input->rankOf());
    const LongType rowLen = input->sizeAt(-1);
    REQUIRE_TRUE(gain->lengthOf() == rowLen, 0,
                 "fused_layer_norm: gain length %lld must equal the last input dimension %lld", gain->lengthOf(),
                 rowLen);
    REQUIRE_TRUE(bias == nullptr || bias->lengthOf() == rowLen, 0,
                 "fused_layer_norm: bias length must equal the last input dimension %lld", rowLen);
    REQUIRE_TRUE(output->isSameShape(input), 0, "fused_layer_norm: output must have the input's shape");

    helpers::fusedLayerNorm(input, gain, bias, output, epsilon, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_layer_norm) {
    return SHAPELIST(denseShapeLike(inputShape->at(0)));
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

    // As in the forward pass: the kernels read and write rows of the last dimension's length and vectors of that
    // length, so any other size reads or writes out of bounds.
    REQUIRE_TRUE(input->rankOf() >= 1, 0, "fused_layer_norm_bp: input must have rank >= 1, got rank %i",
                 input->rankOf());
    const LongType rowLen = input->sizeAt(-1);
    REQUIRE_TRUE(gain->lengthOf() == rowLen, 0,
                 "fused_layer_norm_bp: gain length %lld must equal the last input dimension %lld", gain->lengthOf(),
                 rowLen);
    REQUIRE_TRUE(gradOut->isSameShape(input), 0, "fused_layer_norm_bp: gradient must have the input's shape");
    REQUIRE_TRUE(gradInput->isSameShape(input), 0, "fused_layer_norm_bp: input gradient must have the input's shape");
    REQUIRE_TRUE(gradGain->lengthOf() == rowLen, 0,
                 "fused_layer_norm_bp: gain gradient length must equal the last input dimension %lld", rowLen);
    REQUIRE_TRUE(gradBias == nullptr || gradBias->lengthOf() == rowLen, 0,
                 "fused_layer_norm_bp: bias gradient length must equal the last input dimension %lld", rowLen);

    helpers::fusedLayerNormBackward(input, gain, gradOut, gradInput, gradGain, gradBias,
                                     epsilon, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_layer_norm_bp) {
    // dx, dgain and, with a bias (input 3), dbias: each a dense array shaped like the input it is the gradient of
    // (the type and shape of the gain and the bias are theirs, whatever the input's type)
    auto shapes = SHAPELIST(denseShapeLike(inputShape->at(0)), denseShapeLike(inputShape->at(1)));
    if (inputShape->size() > 3) shapes->push_back(denseShapeLike(inputShape->at(3)));
    return shapes;
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

    // The helpers rotate [batch, seq, heads, head_dim] (or [batch, seq, head_dim]) and write one output element per
    // input element: any other rank or output size reads or writes out of bounds.
    REQUIRE_TRUE(input->rankOf() == 3 || input->rankOf() == 4, 0,
                 "fused_rope: input must be rank 4 [batch, seq, heads, head_dim] or rank 3 [batch, seq, head_dim], "
                 "got rank %i", input->rankOf());
    REQUIRE_TRUE(output->isSameShape(input), 0, "fused_rope: output must have the input's shape");

    if (block.width() >= 3) {
        // Cached path: cos and sin provided as inputs 1 and 2
        auto cosValues = INPUT_VARIABLE(1);
        auto sinValues = INPUT_VARIABLE(2);

        // The helpers read cos and sin at (batch, seq, pair) of the rotation, each table through its own strides: a
        // rank 2 table [seq, half_dim] serves every batch, a rank 3 [batch, seq, half_dim] or rank 4
        // [batch, seq, 1, half_dim] table holds one per batch. A table with less than the rotation reads or a sin
        // table of another shape than the cos table reads out of bounds.
        const int tableRank = cosValues->rankOf();
        REQUIRE_TRUE(tableRank >= 2 && tableRank <= 4, 0,
                     "fused_rope: cos must be rank 2 [seq, half_dim], rank 3 [batch, seq, half_dim] or rank 4 "
                     "[batch, seq, 1, half_dim], got rank %i", tableRank);
        REQUIRE_TRUE(sinValues->isSameShape(cosValues), 0, "fused_rope: sin must have the shape of cos");
        REQUIRE_TRUE(cosValues->sizeAt(tableRank == 2 ? 0 : 1) >= input->sizeAt(1), 0,
                     "fused_rope: cos and sin must hold a row for each of the %lld sequence positions of the input",
                     input->sizeAt(1));
        REQUIRE_TRUE(cosValues->sizeAt(tableRank - 1) >= input->sizeAt(input->rankOf() - 1) / 2, 0,
                     "fused_rope: cos and sin must hold half the head dimension (%lld) per row",
                     input->sizeAt(input->rankOf() - 1) / 2);
        REQUIRE_TRUE(tableRank == 2 || cosValues->sizeAt(0) >= input->sizeAt(0), 0,
                     "fused_rope: cos and sin must hold a table for each of the %lld batch entries of the input",
                     input->sizeAt(0));

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
    return SHAPELIST(denseShapeLike(inputShape->at(0)));
}

DECLARE_TYPES(fused_rope) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
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

    // The helper rotates the gradient's [batch, seq, heads, head_dim] (or [batch, seq, head_dim]) and writes one
    // element of the input gradient per element of it.
    REQUIRE_TRUE(gradOut->rankOf() == 3 || gradOut->rankOf() == 4, 0,
                 "fused_rope_bp: gradient must be rank 4 [batch, seq, heads, head_dim] or rank 3 "
                 "[batch, seq, head_dim], got rank %i", gradOut->rankOf());
    REQUIRE_TRUE(gradIn->isSameShape(gradOut), 0, "fused_rope_bp: input gradient must have the gradient's shape");

    helpers::fusedRoPEBackward(gradOut, gradIn, positionOffset, freqBase, freqScale, ropeType,
                                block.launchContext(), rotaryDimsBp);

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_rope_bp) {
    return SHAPELIST(denseShapeLike(inputShape->at(0)));
}

DECLARE_TYPES(fused_rope_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
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

    // One residual and one output element per input element: the arrays pair up in the flattened, C-order sequence of
    // their elements (so any shapes of the same length do), and the bias repeats along it.
    REQUIRE_TRUE(residual->lengthOf() == input->lengthOf(), 0,
                 "fused_bias_dropout_residual: residual must have as many elements as the input (%lld), got %lld",
                 input->lengthOf(), residual->lengthOf());
    REQUIRE_TRUE(output->lengthOf() == input->lengthOf(), 0,
                 "fused_bias_dropout_residual: output must have as many elements as the input (%lld), got %lld",
                 input->lengthOf(), output->lengthOf());
    REQUIRE_TRUE(bias->lengthOf() > 0, 0, "fused_bias_dropout_residual: bias must not be empty");

    helpers::fusedBiasDropoutResidual(input, bias, residual, output, dropoutProb, seed,
                                       training, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_bias_dropout_residual) {
    return SHAPELIST(denseShapeLike(inputShape->at(0)));
}

DECLARE_TYPES(fused_bias_dropout_residual) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_TERNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
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

    // [batch, seq_len, hidden_dim] normalized by a gamma of hidden_dim elements, then projected by two
    // [hidden_dim, intermediate_dim] weights into [batch, seq_len, intermediate_dim]: any other size reads or writes
    // out of bounds.
    REQUIRE_TRUE(input->rankOf() == 3, 0,
                 "fused_rms_norm_swiglu: input must be rank 3 [batch, seq_len, hidden_dim], got rank %i",
                 input->rankOf());
    const LongType hiddenDim = input->sizeAt(2);
    REQUIRE_TRUE(gamma->lengthOf() == hiddenDim, 0,
                 "fused_rms_norm_swiglu: gamma length %lld must equal the hidden dimension %lld", gamma->lengthOf(),
                 hiddenDim);
    REQUIRE_TRUE(wGate->rankOf() == 2 && wGate->sizeAt(0) == hiddenDim, 0,
                 "fused_rms_norm_swiglu: wGate must be [hidden_dim = %lld, intermediate_dim]", hiddenDim);
    REQUIRE_TRUE(wUp->isSameShape(wGate), 0, "fused_rms_norm_swiglu: wUp must have wGate's shape");
    REQUIRE_TRUE(output->rankOf() == 3 && output->sizeAt(0) == input->sizeAt(0) &&
                     output->sizeAt(1) == input->sizeAt(1) && output->sizeAt(2) == wGate->sizeAt(1),
                 0, "fused_rms_norm_swiglu: output must be [batch, seq_len, intermediate_dim]");

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

    // As in the forward pass, and each gradient has the size of the input it is the gradient of: any other size reads
    // or writes out of bounds.
    REQUIRE_TRUE(input->rankOf() == 3, 0,
                 "fused_rms_norm_swiglu_bp: input must be rank 3 [batch, seq_len, hidden_dim], got rank %i",
                 input->rankOf());
    const LongType hiddenDim = input->sizeAt(2);
    REQUIRE_TRUE(gamma->lengthOf() == hiddenDim, 0,
                 "fused_rms_norm_swiglu_bp: gamma length %lld must equal the hidden dimension %lld", gamma->lengthOf(),
                 hiddenDim);
    REQUIRE_TRUE(wGate->rankOf() == 2 && wGate->sizeAt(0) == hiddenDim, 0,
                 "fused_rms_norm_swiglu_bp: wGate must be [hidden_dim = %lld, intermediate_dim]", hiddenDim);
    REQUIRE_TRUE(wUp->isSameShape(wGate), 0, "fused_rms_norm_swiglu_bp: wUp must have wGate's shape");
    REQUIRE_TRUE(gradOut->rankOf() == 3 && gradOut->sizeAt(0) == input->sizeAt(0) &&
                     gradOut->sizeAt(1) == input->sizeAt(1) && gradOut->sizeAt(2) == wGate->sizeAt(1),
                 0, "fused_rms_norm_swiglu_bp: gradient must be [batch, seq_len, intermediate_dim]");
    REQUIRE_TRUE(gradInput->isSameShape(input), 0, "fused_rms_norm_swiglu_bp: input gradient must have the input's shape");
    REQUIRE_TRUE(gradGamma->lengthOf() == hiddenDim, 0,
                 "fused_rms_norm_swiglu_bp: gamma gradient length must equal the hidden dimension %lld", hiddenDim);
    REQUIRE_TRUE(gradWGate->isSameShape(wGate), 0, "fused_rms_norm_swiglu_bp: wGate gradient must have wGate's shape");
    REQUIRE_TRUE(gradWUp->isSameShape(wUp), 0, "fused_rms_norm_swiglu_bp: wUp gradient must have wUp's shape");

    helpers::fusedRmsNormSwiGLUBackward(input, gamma, wGate, wUp, gradOut,
                                         gradInput, gradGamma, gradWGate, gradWUp,
                                         epsilon, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_rms_norm_swiglu_bp) {
    // dx, dgamma, dwGate and dwUp: each a dense array shaped like the input it is the gradient of, in that input's
    // type (the gamma and the weights may be of another float type than the input)
    return SHAPELIST(denseShapeLike(inputShape->at(0)), denseShapeLike(inputShape->at(1)),
                     denseShapeLike(inputShape->at(2)), denseShapeLike(inputShape->at(3)));
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
  getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// fused_mrope - Multimodal Rotary Position Embedding (M-RoPE)
//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_fused_mrope)
CUSTOM_OP_IMPL(fused_mrope, 4, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto posT = INPUT_VARIABLE(1);
    auto posH = INPUT_VARIABLE(2);
    auto posW = INPUT_VARIABLE(3);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(input->rankOf() == 4, 0,
        "fused_mrope: input must be rank-4 [batch, seq, heads, head_dim], got rank %i", input->rankOf());
    REQUIRE_TRUE(posT->rankOf() == 2, 0,
        "fused_mrope: position_t must be rank-2 [batch, seq], got rank %i", posT->rankOf());

    int sectionT = block.getIArguments()->size() > 0 ? INT_ARG(0) : 24;
    int sectionH = block.getIArguments()->size() > 1 ? INT_ARG(1) : 20;
    int sectionW = block.getIArguments()->size() > 2 ? INT_ARG(2) : 20;
    bool interleaved = block.getIArguments()->size() > 3 ? INT_ARG(3) != 0 : false;
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;

    int headDim = (int) input->sizeAt(3);
    REQUIRE_TRUE(sectionT + sectionH + sectionW == headDim, 0,
        "fused_mrope: sections (%d + %d + %d = %d) must sum to head_dim (%d)",
        sectionT, sectionH, sectionW, sectionT + sectionH + sectionW, headDim);

    // Each rotation pairs an element with the one half a head dimension on, so an odd head dimension would leave its
    // last element of the output unwritten; each (batch, seq) position of the three position tensors rotates the
    // heads of its row of the input (the height and width positions are read in the flattened, C-order sequence of
    // their elements, so any shapes of position_t's length do); the output has an element per input element.
    REQUIRE_TRUE(headDim % 2 == 0, 0, "fused_mrope: head_dim must be even, got %d", headDim);
    REQUIRE_TRUE(posT->sizeAt(0) == input->sizeAt(0) && posT->sizeAt(1) == input->sizeAt(1), 0,
        "fused_mrope: position_t must be [batch, seq] of the input");
    REQUIRE_TRUE(posH->lengthOf() == posT->lengthOf() && posW->lengthOf() == posT->lengthOf(), 0,
        "fused_mrope: position_h and position_w must have as many elements as position_t (%lld)", posT->lengthOf());
    REQUIRE_TRUE(output->isSameShape(input), 0, "fused_mrope: output must have the input's shape");

    helpers::fusedMRoPE(input, posT, posH, posW, output,
                         sectionT, sectionH, sectionW, interleaved, freqBase,
                         block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_mrope) {
    return SHAPELIST(denseShapeLike(inputShape->at(0)));
}

DECLARE_TYPES(fused_mrope) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
}
#endif

//////////////////////////////////////////////////////////////////////////
// vision_embedding_merge - scatter vision embeddings into text at target token positions
//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_vision_embedding_merge)
CUSTOM_OP_IMPL(vision_embedding_merge, 3, 1, false, 0, 1) {
    auto textEmbeddings = INPUT_VARIABLE(0);    // [batch, seqLen, hidden]
    auto visionEmbeddings = INPUT_VARIABLE(1);  // [batch, visionTokens, hidden]
    auto tokenIds = INPUT_VARIABLE(2);          // [batch, seqLen]
    auto output = OUTPUT_VARIABLE(0);           // [batch, seqLen, hidden]

    sd::LongType targetTokenId = INT_ARG(0);

    REQUIRE_TRUE(textEmbeddings->rankOf() == 3, 0,
        "vision_embedding_merge: textEmbeddings must be rank 3, got %i", textEmbeddings->rankOf());
    REQUIRE_TRUE(visionEmbeddings->rankOf() == 3, 0,
        "vision_embedding_merge: visionEmbeddings must be rank 3, got %i", visionEmbeddings->rankOf());
    REQUIRE_TRUE(tokenIds->rankOf() == 2, 0,
        "vision_embedding_merge: tokenIds must be rank 2, got %i", tokenIds->rankOf());
    REQUIRE_TRUE(textEmbeddings->sizeAt(2) == visionEmbeddings->sizeAt(2), 0,
        "vision_embedding_merge: hidden dim mismatch: text=%lld vs vision=%lld",
        textEmbeddings->sizeAt(2), visionEmbeddings->sizeAt(2));

    helpers::visionEmbeddingMerge(textEmbeddings, visionEmbeddings, tokenIds,
                                  output, targetTokenId, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(vision_embedding_merge) {
    // Output shape is the same as textEmbeddings: [batch, seqLen, hidden], a fresh array on dense strides (the kernels
    // write it through its own strides, so strides inherited from a view of textEmbeddings would run past its buffer)
    return SHAPELIST(denseShapeLike(inputShape->at(0)));
}

DECLARE_TYPES(vision_embedding_merge) {
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
    getOpDescriptor()->setAllowedInputTypes(1, {ALL_FLOATS});
    getOpDescriptor()->setAllowedInputTypes(2, {ALL_INTS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

}  // namespace ops
}  // namespace sd

#endif
