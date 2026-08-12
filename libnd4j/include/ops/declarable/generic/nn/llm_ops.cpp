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
// Generic (fallback) implementations for LLM operations.
// These are used when platform-specific helpers (GGML/llama.cpp) are not available.
//

#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>
#if NOT_EXCLUDED(OP_rms_norm) || NOT_EXCLUDED(OP_rope) || NOT_EXCLUDED(OP_silu) || \
    NOT_EXCLUDED(OP_quantized_matmul) || NOT_EXCLUDED(OP_grouped_query_attention) || \
    NOT_EXCLUDED(OP_flash_attention) || NOT_EXCLUDED(OP_kv_cache_update) || \
    NOT_EXCLUDED(OP_apply_alibi) || NOT_EXCLUDED(OP_sliding_window_attention) || \
    NOT_EXCLUDED(OP_swish_mul) || NOT_EXCLUDED(OP_mean_square) || \
    NOT_EXCLUDED(OP_column_parallel_linear) || NOT_EXCLUDED(OP_row_parallel_linear) || \
    NOT_EXCLUDED(OP_kv_cache_quantize) || NOT_EXCLUDED(OP_kv_cache_dequantize) || \
    NOT_EXCLUDED(OP_ggml_dequantize) || NOT_EXCLUDED(OP_fused_gemm_swiglu) || \
    NOT_EXCLUDED(OP_rms_norm_linear) || NOT_EXCLUDED(OP_skip_rms_norm)

#include <ops/declarable/headers/llm.h>
#include <helpers/MmulHelper.h>
#include <helpers/FlashAttentionHelper.h>
#include <ops/declarable/helpers/rms_norm.h>
#include <ops/declarable/helpers/fused_llm_ops.h>
#include <helpers/ShapeUtils.h>
#include <math/templatemath.h>
#include <execution/Threads.h>
#include <array/DataTypeUtils.h>
#include <ops/declarable/helpers/kv_cache_quantize.h>
#include <ops/declarable/helpers/ggml_dequantize.h>
#include <cmath>
#include <cstring>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// rms_norm - Root Mean Square Layer Normalization
#if NOT_EXCLUDED(OP_rms_norm)
CUSTOM_OP_IMPL(rms_norm, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    NDArray* gamma = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;
    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    // Fast path conditions: last-dim normalization with contiguous row-major data
    const int rank = input->rankOf();
    const bool inputContiguous = input->ordering() == 'c' &&
                                 shape::strideDescendingCAscendingF(input->shapeInfo());
    const bool outputContiguous = output->ordering() == 'c' &&
                                  shape::strideDescendingCAscendingF(output->shapeInfo());
    const bool isContiguous = inputContiguous && outputContiguous;
    const bool isFloat = input->dataType() == DataType::FLOAT32;
    const bool isDouble = input->dataType() == DataType::DOUBLE;
    const bool isHalf = input->dataType() == DataType::HALF;
    const bool gammaContiguous = gamma == nullptr ||
                                 shape::strideDescendingCAscendingF(gamma->shapeInfo());

    // Fast path: fused helper (linker resolves CPU vs CUDA impl)
    if ((isFloat || isDouble || isHalf) && gammaContiguous) {
        const NDArray* inputToUse = input;
        NDArray* contiguousInput = nullptr;
        if (!inputContiguous) {
            contiguousInput = new NDArray(input->dup('c'));
            inputToUse = contiguousInput;
        }
        NDArray* outputToUse = output;
        NDArray* contiguousOutput = nullptr;
        if (!outputContiguous) {
            contiguousOutput = new NDArray(output->dup('c'));
            outputToUse = contiguousOutput;
        }

        // The CUDA kernel now accepts gamma in its native dtype via dual-type
        // template instantiations (e.g., <float16, float> for F16 input + F32 gamma).
        // No gamma cast needed — eliminates one transformAnySimpleCached kernel per call.
        helpers::rmsNorm(block.launchContext(), const_cast<NDArray*>(inputToUse), gamma, outputToUse, eps);

        if (contiguousOutput != nullptr) {
            output->assign(contiguousOutput);
            delete contiguousOutput;
        }
        if (contiguousInput != nullptr) {
            delete contiguousInput;
        }
        return Status::OK;
    }

    // General fallback path for unsupported dtypes
    std::vector<LongType> axis = {input->rankOf() - 1};
    NDArray* squared = (*input) * (*input);
    NDArray* meanSquared = squared->reduceAlongDimension(reduce::Mean, &axis, true);
    delete squared;

    NDArray* meanPlusEps = (*meanSquared) + eps;
    delete meanSquared;
    NDArray* rsqrt = meanPlusEps->transform(transform::RSqrt);
    delete meanPlusEps;

    NDArray* result = (*input) * (*rsqrt);
    output->assign(result);
    delete result;
    delete rsqrt;

    if (gamma != nullptr) {
        output->applyBroadcast(broadcast::Multiply, &axis, gamma, output);
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(rms_norm) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(rms_norm) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
}

// rms_norm_bp: inputs are [input, gradOut] or [input, gradOut, gamma].
// When gamma is present (block.width() >= 3), outputs are [gradIn, gradGamma].
// When gamma is absent, output is [gradIn] only.
CUSTOM_OP_IMPL(rms_norm_bp, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gradOut = INPUT_VARIABLE(1);
    NDArray* gamma = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto gradIn = OUTPUT_VARIABLE(0);
    NDArray* gradGamma = (gamma != nullptr && block.outputWidth() > 1) ? OUTPUT_VARIABLE(1) : nullptr;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    // For HALF/BFLOAT16, rsqrt^3 can overflow even when the mathematically expected
    // result is finite, and zero * inf then poisons gradients with NaN. Keep the
    // backward intermediates in at least FLOAT32 and cast/assign at the outputs.
    bool inputLowPrecision = input->dataType() == DataType::HALF || input->dataType() == DataType::BFLOAT16;
    bool gradLowPrecision = gradOut->dataType() == DataType::HALF || gradOut->dataType() == DataType::BFLOAT16;
    bool gammaLowPrecision = gamma != nullptr &&
        (gamma->dataType() == DataType::HALF || gamma->dataType() == DataType::BFLOAT16);
    DataType calcType = input->dataType() == DataType::DOUBLE ? DataType::DOUBLE : DataType::FLOAT32;
    bool useCalcCast = inputLowPrecision || gradLowPrecision || gammaLowPrecision;

    NDArray* inputCalc = input;
    NDArray* gradOutCalc = gradOut;
    NDArray* gammaCalc = gamma;
    NDArray* inputCast = nullptr;
    NDArray* gradOutCast = nullptr;
    NDArray* gammaCast = nullptr;

    if (useCalcCast) {
        if (input->dataType() != calcType) {
            inputCast = input->cast(calcType);
            inputCalc = inputCast;
        }
        if (gradOut->dataType() != calcType) {
            gradOutCast = gradOut->cast(calcType);
            gradOutCalc = gradOutCast;
        }
        if (gamma != nullptr && gamma->dataType() != calcType) {
            gammaCast = gamma->cast(calcType);
            gammaCalc = gammaCast;
        }
    }

    std::vector<LongType> axis = {inputCalc->rankOf() - 1};
    auto n = inputCalc->sizeAt(axis[0]);

    // Forward pass values
    NDArray* squared = (*inputCalc) * (*inputCalc);
    NDArray* meanSquared = squared->reduceAlongDimension(reduce::Mean, &axis, true);
    delete squared;

    NDArray* meanPlusEps = (*meanSquared) + eps;
    delete meanSquared;
    NDArray* rsqrt = meanPlusEps->transform(transform::RSqrt);
    delete meanPlusEps;

    // Effective upstream gradient: when gamma is present, gradOut is after the gamma scaling.
    // We need to chain through gamma: effective_gradOut_for_norm = gradOut * gamma (broadcast).
    NDArray* effectiveGradOut = nullptr;
    bool ownedEffectiveGradOut = false;
    if (gammaCalc != nullptr) {
        // gradOut has the same shape as the normed output [batch..., features].
        // gamma has shape [features]. Multiply element-wise via broadcast.
        effectiveGradOut = (*gradOutCalc) * (*gammaCalc);
        ownedEffectiveGradOut = true;
    } else {
        effectiveGradOut = gradOutCalc;
    }

    // Gradient computation for input
    // dL/dx = effectiveGradOut * rsqrt - input * (dot(input, effectiveGradOut)/n) * rsqrt^3
    NDArray* gradNorm = (*effectiveGradOut) * (*rsqrt);
    NDArray* inputGrad = (*inputCalc) * (*effectiveGradOut);
    NDArray* dotProduct = inputGrad->reduceAlongDimension(reduce::Sum, &axis, true);
    delete inputGrad;

    NDArray* rsqrt2 = (*rsqrt) * (*rsqrt);
    NDArray* rsqrt3 = (*rsqrt2) * (*rsqrt);
    delete rsqrt2;

    NDArray* gradMean = (*dotProduct) * (*rsqrt3);
    delete dotProduct;
    delete rsqrt3;

    NDArray* gradMeanScaled = (*gradMean) / static_cast<float>(n);
    delete gradMean;

    NDArray* inputScaled = (*inputCalc) * (*gradMeanScaled);
    delete gradMeanScaled;

    NDArray* result = (*gradNorm) - (*inputScaled);
    delete gradNorm;
    delete inputScaled;

    gradIn->assign(result);
    delete result;

    if (ownedEffectiveGradOut) {
        delete effectiveGradOut;
    }

    // Gradient for gamma: dL/dgamma = sum_over_batch_dims(gradOut * x_normed)
    // x_normed = input * rsqrt (before gamma scaling). rsqrt is still alive here.
    if (gradGamma != nullptr) {
        // Compute x_normed = input * rsqrt
        NDArray* xNormed = (*inputCalc) * (*rsqrt);
        // dL/dgamma = sum(gradOut * x_normed) over all dims except the last (features) dim
        NDArray* gammaGradFull = (*gradOutCalc) * (*xNormed);
        delete xNormed;
        // Sum over all leading dimensions (all except last)
        std::vector<LongType> batchDims;
        for (int d = 0; d < inputCalc->rankOf() - 1; d++) batchDims.push_back(d);
        NDArray* gammaGradReduced = gammaGradFull->reduceAlongDimension(reduce::Sum, &batchDims, false);
        delete gammaGradFull;
        gradGamma->assign(gammaGradReduced);
        delete gammaGradReduced;
    }

    delete rsqrt;
    delete inputCast;
    delete gradOutCast;
    delete gammaCast;

    return Status::OK;
}

DECLARE_SHAPE_FN(rms_norm_bp) {
    auto inShape = inputShape->at(0);
    auto gradInShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary();
    if (inputShape->size() > 2) {
        // gamma present at input 2: output both gradIn and gradGamma
        auto gammaShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(2))->primary();
        return SHAPELIST(gradInShape, gammaShape);
    }
    return SHAPELIST(gradInShape);
}

DECLARE_TYPES(rms_norm_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// skip_rms_norm - Fused Residual Add + RMS Normalization
#if NOT_EXCLUDED(OP_skip_rms_norm)
CUSTOM_OP_IMPL(skip_rms_norm, 3, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto skip = INPUT_VARIABLE(1);
    auto gamma = INPUT_VARIABLE(2);
    NDArray* bias = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

    auto output = OUTPUT_VARIABLE(0);
    NDArray* hiddenOut = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    // The CUDA kernel now accepts gamma in its native dtype via dual-type
    // template instantiations (e.g., <float16, float> for F16 input + F32 gamma).
    // No gamma cast needed — eliminates one transformAnySimpleCached kernel per call.
    helpers::skipRmsNorm(block.launchContext(), input, skip, gamma, bias, output, hiddenOut, eps);

    return Status::OK;
}

DECLARE_SHAPE_FN(skip_rms_norm) {
    auto inShape = inputShape->at(0);
    auto outShapes = SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());

    // Second output (pre-norm hidden states) has same shape as input, only if graph requests it
    if (block.outputWidth() > 1) {
        outShapes->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
    }

    return outShapes;
}

DECLARE_TYPES(skip_rms_norm) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
}
#endif

//////////////////////////////////////////////////////////////////////////
// rope - Rotary Position Embedding
#if NOT_EXCLUDED(OP_rope)
CUSTOM_OP_IMPL(rope, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);  // [batch, seq_len, num_heads, head_dim]
    auto output = OUTPUT_VARIABLE(0);

    // rope arg order matches RoPE.java: INT_ARG(0)=mode/ropeType, INT_ARG(1)=nPast/positionOffset, INT_ARG(2)=nDims/rotaryDims
    int ropeType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    LongType positionOffset = block.getIArguments()->size() > 1 ? static_cast<LongType>(INT_ARG(1)) : 0;
    int rotaryDims = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

    // fusedRoPE expects an NDArray* for position — wrap the scalar offset
    auto posArr = NDArrayFactory::create<LongType>(positionOffset, block.launchContext());
    helpers::fusedRoPE(input, output, posArr, freqBase, freqScale, ropeType,
                       block.launchContext(), rotaryDims);
    delete posArr;

    return Status::OK;
}

DECLARE_SHAPE_FN(rope) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(rope) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
}

CUSTOM_OP_IMPL(rope_bp, 2, 1, false, 0, 0) {
    auto gradOut = INPUT_VARIABLE(1);
    auto gradIn = OUTPUT_VARIABLE(0);

    // rope_bp arg order matches rope: INT_ARG(0)=mode/ropeType, INT_ARG(1)=nPast/positionOffset, INT_ARG(2)=nDims/rotaryDims
    int ropeType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int positionOffset = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int rotaryDims = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

    // Delegate to the platform-correct backward helper
    helpers::fusedRoPEBackward(gradOut, gradIn, positionOffset, freqBase, freqScale, ropeType,
                               block.launchContext(), rotaryDims);

    return Status::OK;
}

DECLARE_SHAPE_FN(rope_bp) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(rope_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// silu - SiLU/Swish Activation
#if NOT_EXCLUDED(OP_silu)
CONFIGURABLE_OP_IMPL(silu, 1, 1, true, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // silu(x) = x * sigmoid(x)
    if (output->buffer() == input->buffer()) {
        // In-place: output IS input — sigmoid(x) would destroy x before multiply.
        // Use temp allocation for correctness.
        NDArray* sigmoid = input->transform(transform::Sigmoid);
        output->applyPairwiseTransform(pairwise::Multiply, sigmoid, output);
        delete sigmoid;
    } else {
        // Out-of-place: write sigmoid into output, then multiply by input.
        // Eliminates 2 temporary allocations + 1 Assign copy kernel.
        input->applyTransform(transform::Sigmoid, output);
        output->applyPairwiseTransform(pairwise::Multiply, input, output);
    }

    return Status::OK;
}

DECLARE_TYPES(silu) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION);
}

CONFIGURABLE_OP_IMPL(silu_bp, 2, 1, true, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gradOut = INPUT_VARIABLE(1);
    auto gradIn = OUTPUT_VARIABLE(0);

    // d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    NDArray* sigmoid = input->transform(transform::Sigmoid);

    // (1 - sigmoid)
    NDArray* oneMinusSigmoid = (*sigmoid) * (-1.0f);
    NDArray* oneMinusSigmoid2 = (*oneMinusSigmoid) + 1.0f;
    delete oneMinusSigmoid;

    // sigmoid * (1 - sigmoid)
    NDArray* sigmoidDeriv = (*sigmoid) * (*oneMinusSigmoid2);
    delete oneMinusSigmoid2;

    // x * sigmoid'
    NDArray* xSigmoidDeriv = (*input) * (*sigmoidDeriv);
    delete sigmoidDeriv;

    // sigmoid + x * sigmoid'
    NDArray* grad = (*sigmoid) + (*xSigmoidDeriv);
    delete sigmoid;
    delete xSigmoidDeriv;

    // gradOut * grad
    NDArray* result = (*gradOut) * (*grad);
    delete grad;

    gradIn->assign(result);
    delete result;

    return Status::OK;
}

DECLARE_TYPES(silu_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// quantized_matmul - Quantized Matrix Multiplication
#if NOT_EXCLUDED(OP_quantized_matmul)
CUSTOM_OP_IMPL(quantized_matmul, 2, 1, false, 0, 0) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    // For generic implementation, just do regular matmul
    // Platform helpers will do actual quantized computation
    MmulHelper::mmul(a, b, c, 1.0f, 0.0f);

    return Status::OK;
}

DECLARE_SHAPE_FN(quantized_matmul) {
    auto aShape = inputShape->at(0);
    auto bShape = inputShape->at(1);
    auto dtype = ArrayOptions::dataType(aShape);

    auto M = shape::sizeAt(aShape, static_cast<LongType>(0));
    auto K = shape::sizeAt(aShape, static_cast<LongType>(1));
    auto N = shape::sizeAt(bShape, static_cast<LongType>(1));

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {M, N}));
}

DECLARE_TYPES(quantized_matmul) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

// llama.cpp-compat name
DECLARE_SYN(quantized_mul_mat, quantized_matmul);
#endif

//////////////////////////////////////////////////////////////////////////
// grouped_query_attention - Grouped Query Attention using FlashAttentionHelper
#if NOT_EXCLUDED(OP_grouped_query_attention)
CUSTOM_OP_IMPL(grouped_query_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);   // [batch, seq_len, num_heads, head_dim]
    auto key = INPUT_VARIABLE(1);     // [batch, kv_len, num_kv_heads, head_dim]
    auto value = INPUT_VARIABLE(2);   // [batch, kv_len, num_kv_heads, head_dim]
    auto output = OUTPUT_VARIABLE(0);

    // Parse configuration
    // T_ARG: scale
    // B_ARG: isCausal
    // I_ARG: numHeads, numKvHeads
    FlashAttentionHelper::Config config;
    config.scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;
    config.isCausal = block.numB() > 0 ? B_ARG(0) : true;
    config.numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : query->sizeAt(2);
    config.numKvHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : config.numHeads;

    // Use FlashAttentionHelper which handles GQA automatically
    FlashAttentionHelper::forward(query, key, value, output, config, nullptr, nullptr, nullptr, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(grouped_query_attention) {
    auto queryShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(queryShape)->primary());
}

DECLARE_TYPES(grouped_query_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}

// llama.cpp-compat name (head counts derive from input shapes when iArgs are absent)
DECLARE_SYN(gqa_attention, grouped_query_attention);

CUSTOM_OP_IMPL(grouped_query_attention_bp, 4, 3, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto gradOutput = INPUT_VARIABLE(3);

    auto gradQ = OUTPUT_VARIABLE(0);
    auto gradK = OUTPUT_VARIABLE(1);
    auto gradV = OUTPUT_VARIABLE(2);

    // Parse configuration
    // T_ARG: scale
    // B_ARG: isCausal
    // I_ARG: numHeads, numKvHeads
    FlashAttentionHelper::Config config;
    config.scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;
    config.isCausal = block.numB() > 0 ? B_ARG(0) : true;
    config.numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : query->sizeAt(2);
    config.numKvHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : config.numHeads;

    // Compute forward pass to get output and LSE for backward
    auto queryShape = query->getShapeAsVector();
    auto computedOutput = NDArrayFactory::create_('c', *queryShape, query->dataType(), block.launchContext());
    delete queryShape;
    auto seqLen = query->sizeAt(1);
    auto numHeads = query->sizeAt(2);
    auto batch = query->sizeAt(0);
    std::vector<sd::LongType> lseShape = {batch, numHeads, seqLen};
    auto computedLse = NDArrayFactory::create_('c', lseShape, query->dataType(), block.launchContext());

    FlashAttentionHelper::forward(query, key, value, computedOutput, config, computedLse, nullptr, nullptr, block.launchContext());

    // Run backward pass
    FlashAttentionHelper::backward(gradOutput, query, key, value, computedOutput, computedLse,
                                   gradQ, gradK, gradV, config, block.launchContext());

    delete computedOutput;
    delete computedLse;

    return Status::OK;
}

DECLARE_SHAPE_FN(grouped_query_attention_bp) {
    auto queryShape = inputShape->at(0);
    auto keyShape = inputShape->at(1);
    auto valueShape = inputShape->at(2);

    return SHAPELIST(
        ConstantShapeHelper::getInstance().bufferForShapeInfo(queryShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(keyShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(valueShape)->primary());
}

DECLARE_TYPES(grouped_query_attention_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// flash_attention - Memory-efficient attention using FlashAttentionHelper
#if NOT_EXCLUDED(OP_flash_attention)
CUSTOM_OP_IMPL(flash_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    // Parse configuration
    // T_ARG: scale
    // B_ARG: isCausal
    // I_ARG: numHeads, numKvHeads
    FlashAttentionHelper::Config config;
    config.scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;  // 0 = auto-compute scale
    config.isCausal = block.numB() > 0 ? B_ARG(0) : true;
    config.numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : query->sizeAt(2);
    config.numKvHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : key->sizeAt(2);

    // Use FlashAttentionHelper for memory-efficient computation
    FlashAttentionHelper::forward(query, key, value, output, config, nullptr, nullptr, nullptr, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(flash_attention) {
    auto queryShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(queryShape)->primary());
}

DECLARE_TYPES(flash_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}

CUSTOM_OP_IMPL(flash_attention_bp, 4, 3, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto gradOutput = INPUT_VARIABLE(3);

    // Optional inputs for backward pass
    NDArray* output = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
    NDArray* softmaxLse = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;

    auto gradQ = OUTPUT_VARIABLE(0);
    auto gradK = OUTPUT_VARIABLE(1);
    auto gradV = OUTPUT_VARIABLE(2);

    // Parse configuration
    // T_ARG: scale
    // B_ARG: isCausal
    // I_ARG: numHeads, numKvHeads
    FlashAttentionHelper::Config config;
    config.scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;
    config.isCausal = block.numB() > 0 ? B_ARG(0) : true;
    config.numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : query->sizeAt(2);
    config.numKvHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : key->sizeAt(2);

    // If output and softmaxLse not provided, we need to recompute forward pass
    NDArray* computedOutput = nullptr;
    NDArray* computedLse = nullptr;

    if (output == nullptr || softmaxLse == nullptr) {
        // Allocate temporary arrays for forward pass results
        auto queryShapeVec = query->getShapeAsVector();
        computedOutput = NDArrayFactory::create_('c', *queryShapeVec, query->dataType(), block.launchContext());
        delete queryShapeVec;
        auto seqLen = query->sizeAt(1);
        auto numHeads = query->sizeAt(2);
        auto batch = query->sizeAt(0);
        std::vector<sd::LongType> lseShapeVec = {batch, numHeads, seqLen};
        computedLse = NDArrayFactory::create_('c', lseShapeVec, query->dataType(), block.launchContext());

        // Run forward pass to get output and LSE
        FlashAttentionHelper::forward(query, key, value, computedOutput, config, computedLse, nullptr, nullptr, block.launchContext());

        output = computedOutput;
        softmaxLse = computedLse;
    }

    // Run backward pass
    FlashAttentionHelper::backward(gradOutput, query, key, value, output, softmaxLse,
                                   gradQ, gradK, gradV, config, block.launchContext());

    // Cleanup temporary arrays
    if (computedOutput) delete computedOutput;
    if (computedLse) delete computedLse;

    return Status::OK;
}

DECLARE_SHAPE_FN(flash_attention_bp) {
    auto queryShape = inputShape->at(0);
    auto keyShape = inputShape->at(1);
    auto valueShape = inputShape->at(2);

    return SHAPELIST(
        ConstantShapeHelper::getInstance().bufferForShapeInfo(queryShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(keyShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(valueShape)->primary());
}

DECLARE_TYPES(flash_attention_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// kv_cache_update - KV Cache Update
#if NOT_EXCLUDED(OP_kv_cache_update)
CUSTOM_OP_IMPL(kv_cache_update, 4, 2, false, 0, 0) {
    auto keyCache = INPUT_VARIABLE(0);
    auto valueCache = INPUT_VARIABLE(1);
    auto newKeys = INPUT_VARIABLE(2);
    auto newValues = INPUT_VARIABLE(3);

    auto outputKeyCache = OUTPUT_VARIABLE(0);
    auto outputValueCache = OUTPUT_VARIABLE(1);

    int startPos = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    // Copy existing cache
    outputKeyCache->assign(keyCache);
    outputValueCache->assign(valueCache);

    // Update with new keys/values at position using typed buffer copy
    auto newSeqLen = newKeys->sizeAt(1);
    auto batch = newKeys->sizeAt(0);
    auto numHeads = newKeys->rankOf() > 2 ? newKeys->sizeAt(2) : 1;
    auto headDim = newKeys->rankOf() > 3 ? newKeys->sizeAt(3) : newKeys->sizeAt(-1);
    auto cacheSeqLen = keyCache->sizeAt(1);

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; ++b) {
            for (LongType i = 0; i < newSeqLen; ++i) {
                for (LongType h = 0; h < numHeads; ++h) {
                    LongType srcBase = ((b * newSeqLen + i) * numHeads + h) * headDim;
                    LongType dstBase = ((b * cacheSeqLen + startPos + i) * numHeads + h) * headDim;
                    std::memcpy(
                        outputKeyCache->bufferWithOffset(dstBase),
                        newKeys->bufferWithOffset(srcBase),
                        headDim * DataTypeUtils::sizeOfElement(newKeys->dataType()));
                    std::memcpy(
                        outputValueCache->bufferWithOffset(dstBase),
                        newValues->bufferWithOffset(srcBase),
                        headDim * DataTypeUtils::sizeOfElement(newValues->dataType()));
                }
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, batch, 1);

    return Status::OK;
}

DECLARE_SHAPE_FN(kv_cache_update) {
    auto keyCacheShape = inputShape->at(0);
    auto valueCacheShape = inputShape->at(1);

    return SHAPELIST(
        ConstantShapeHelper::getInstance().bufferForShapeInfo(keyCacheShape)->primary(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(valueCacheShape)->primary());
}

DECLARE_TYPES(kv_cache_update) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_VALUE_DEPENDENT_SHAPE);
}
#endif

//////////////////////////////////////////////////////////////////////////
// apply_alibi - ALiBi Position Encoding
#if NOT_EXCLUDED(OP_apply_alibi)
CUSTOM_OP_IMPL(apply_alibi, 1, 1, false, 0, 0) {
    auto scores = INPUT_VARIABLE(0);  // [batch, num_heads, seq_len, kv_len]
    auto output = OUTPUT_VARIABLE(0);

    int numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : static_cast<int>(scores->sizeAt(1));

    output->assign(scores);

    auto batch = scores->sizeAt(0);
    auto seqLen = scores->sizeAt(2);
    auto kvLen = scores->sizeAt(3);

    // Compute ALiBi slopes
    std::vector<float> slopes(numHeads);
    float base = std::pow(2.0f, -8.0f / numHeads);
    for (int h = 0; h < numHeads; ++h) {
        slopes[h] = std::pow(base, h + 1);
    }

    // Apply ALiBi bias — uses e<>/p<> for type-safe FP16/BF16 handling
    // This is a one-time prefill op (not per-token), so accessor overhead is acceptable
    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; ++b) {
            for (int h = 0; h < numHeads; ++h) {
                float slope = slopes[h];
                for (LongType sq = 0; sq < seqLen; ++sq) {
                    LongType rowBase = ((b * numHeads + h) * seqLen + sq) * kvLen;
                    PRAGMA_OMP_SIMD
                    for (LongType sk = 0; sk < kvLen; ++sk) {
                        double bias = -slope * std::abs(static_cast<double>(sq) - static_cast<double>(sk));
                        // Read, add bias, write back via NDArray (type-safe)
                        LongType flatIdx = rowBase + sk;
                        double val = output->e<double>(flatIdx);
                        output->p<double>(flatIdx, val + bias);
                    }
                }
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, batch, 1);

    return Status::OK;
}

DECLARE_SHAPE_FN(apply_alibi) {
    auto scoresShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(scoresShape)->primary());
}

DECLARE_TYPES(apply_alibi) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}

// llama.cpp-compat name (uses standard per-head ALiBi slopes; the old llamacpp
// scalar-slope T arg is intentionally not honored)
DECLARE_SYN(alibi_position_bias, apply_alibi);
#endif

//////////////////////////////////////////////////////////////////////////
// sliding_window_attention - Sliding Window Attention using FlashAttentionHelper
#if NOT_EXCLUDED(OP_sliding_window_attention)
CUSTOM_OP_IMPL(sliding_window_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    // Parse configuration
    FlashAttentionHelper::Config config;
    config.windowSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 4096;
    config.numHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : query->sizeAt(2);
    config.numKvHeads = block.getIArguments()->size() > 2 ? INT_ARG(2) : config.numHeads;
    config.scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;
    config.isCausal = true;  // Sliding window is typically causal

    // Use FlashAttentionHelper which handles sliding window via windowSize config
    FlashAttentionHelper::forward(query, key, value, output, config, nullptr, nullptr, nullptr, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(sliding_window_attention) {
    auto queryShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(queryShape)->primary());
}

DECLARE_TYPES(sliding_window_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}
#endif

//////////////////////////////////////////////////////////////////////////
// swish_mul - SwiGLU component: swish(x) * y
#if NOT_EXCLUDED(OP_swish_mul)
CONFIGURABLE_OP_IMPL(swish_mul, 2, 1, true, 0, 0) {
    auto x = INPUT_VARIABLE(0);  // Input for swish activation
    auto y = INPUT_VARIABLE(1);  // Gate tensor
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(x->dataType() == output->dataType(), 0,
        "swish_mul: input x dtype (%s) != output dtype (%s). "
        "x shape=%s y dtype=%s y shape=%s output shape=%s",
        DataTypeUtils::asString(x->dataType()).c_str(),
        DataTypeUtils::asString(output->dataType()).c_str(),
        ShapeUtils::shapeAsString(x).c_str(),
        DataTypeUtils::asString(y->dataType()).c_str(),
        ShapeUtils::shapeAsString(y).c_str(),
        ShapeUtils::shapeAsString(output).c_str());

    // swish_mul(x, y) = silu(x) * y = x * sigmoid(x) * y
    if (output->buffer() == x->buffer()) {
        // In-place on x: sigmoid(x) would destroy original x.
        // Compute sigmoid into temp, multiply x by sigmoid in-place, then multiply by y.
        NDArray* sigmoid = x->transform(transform::Sigmoid);
        output->applyPairwiseTransform(pairwise::Multiply, sigmoid, output); // output = x * sigmoid(x)
        output->applyPairwiseTransform(pairwise::Multiply, y, output);      // output *= y
        delete sigmoid;
    } else if (output->buffer() == y->buffer()) {
        // In-place on y: write sigmoid(x) into a temp, compute silu(x) into temp,
        // then multiply by y (which is output, still has original value).
        NDArray* sigmoid = x->transform(transform::Sigmoid);
        NDArray* silu = (*x) * (*sigmoid);
        delete sigmoid;
        output->applyPairwiseTransform(pairwise::Multiply, silu, output); // output = y * silu(x)
        delete silu;
    } else {
        // Out-of-place: write sigmoid(x) into output, multiply x, multiply y.
        // Eliminates 3 temporary allocations + 1 Assign copy kernel.
        x->applyTransform(transform::Sigmoid, output);
        output->applyPairwiseTransform(pairwise::Multiply, x, output);
        output->applyPairwiseTransform(pairwise::Multiply, y, output);
    }

    return Status::OK;
}

DECLARE_TYPES(swish_mul) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

CONFIGURABLE_OP_IMPL(swish_mul_bp, 3, 2, true, 0, 0) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto gradOut = INPUT_VARIABLE(2);
    auto gradX = OUTPUT_VARIABLE(0);
    auto gradY = OUTPUT_VARIABLE(1);

    // Forward: out = silu(x) * y = x * sigmoid(x) * y
    // Let s = sigmoid(x), so out = x * s * y
    //
    // d(out)/dx = y * (s + x * s * (1 - s)) = y * s * (1 + x * (1 - s))
    // d(out)/dy = x * s = silu(x)

    NDArray* sigmoid = x->transform(transform::Sigmoid);

    // Gradient w.r.t. y: gradY = gradOut * silu(x) = gradOut * x * sigmoid(x)
    NDArray* silu = (*x) * (*sigmoid);
    NDArray* gradYResult = (*gradOut) * (*silu);
    gradY->assign(gradYResult);
    delete gradYResult;

    // Gradient w.r.t. x: gradX = gradOut * y * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    // = gradOut * y * (sigmoid + x * sigmoid * (1 - sigmoid))
    NDArray* oneMinusSigmoid = (*sigmoid) * (-1.0f);
    NDArray* oneMinusSigmoid2 = (*oneMinusSigmoid) + 1.0f;
    delete oneMinusSigmoid;

    NDArray* sigmoidDeriv = (*sigmoid) * (*oneMinusSigmoid2);  // sigmoid * (1 - sigmoid)
    delete oneMinusSigmoid2;

    NDArray* xSigmoidDeriv = (*x) * (*sigmoidDeriv);  // x * sigmoid * (1 - sigmoid)
    delete sigmoidDeriv;

    NDArray* siluDeriv = (*sigmoid) + (*xSigmoidDeriv);  // sigmoid + x * sigmoid * (1 - sigmoid)
    delete sigmoid;
    delete xSigmoidDeriv;
    delete silu;

    NDArray* gradXTemp = (*gradOut) * (*y);
    NDArray* gradXResult = (*gradXTemp) * (*siluDeriv);
    delete gradXTemp;
    delete siluDeriv;

    gradX->assign(gradXResult);
    delete gradXResult;

    return Status::OK;
}

DECLARE_TYPES(swish_mul_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// fused_gemm_swiglu - Fused GatedMLP: silu(X @ W_gate) * (X @ W_up)
// Eliminates one read of X from HBM by computing both GEMMs on the same input.
// For now, implemented as two matmuls + swish_mul. The CUDA kernel fusion
// (concatenated GEMM) will be added as a platform-specific op.
#if NOT_EXCLUDED(OP_fused_gemm_swiglu)
CUSTOM_OP_IMPL(fused_gemm_swiglu, 3, 1, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);       // [..., K] input activations (rank >= 2)
    auto wGate = INPUT_VARIABLE(1);   // [K, N] gate weight matrix
    auto wUp = INPUT_VARIABLE(2);     // [K, N] up weight matrix
    auto output = OUTPUT_VARIABLE(0); // [..., N]

    REQUIRE_TRUE(x->rankOf() >= 2, 0, "fused_gemm_swiglu: input must be rank >= 2, got %d", x->rankOf());
    REQUIRE_TRUE(wGate->rankOf() == 2, 0, "fused_gemm_swiglu: wGate must be rank 2, got %d", wGate->rankOf());
    REQUIRE_TRUE(wUp->rankOf() == 2, 0, "fused_gemm_swiglu: wUp must be rank 2, got %d", wUp->rankOf());

    auto K = x->sizeAt(-1);
    auto N = wGate->sizeAt(1);

    // For rank > 2, flatten leading dims into M so we can do rank-2 matmul
    NDArray* xFlat = nullptr;
    sd::LongType M = 1;
    bool needReshape = x->rankOf() > 2;
    if (needReshape) {
        for (int d = 0; d < x->rankOf() - 1; d++)
            M *= x->sizeAt(d);
        std::vector<sd::LongType> flatShape = {M, K};
        auto xReshaped = x->reshape('c', flatShape, false);
        xFlat = xReshaped;
    } else {
        xFlat = x;
        M = x->sizeAt(0);
    }

    // Allocate C-order temporaries matching flat shape [M, N]
    std::vector<sd::LongType> outFlatShape = {M, N};
    auto gate = NDArrayFactory::create_('c', outFlatShape, output->dataType(), block.launchContext());
    auto up   = NDArrayFactory::create_('c', outFlatShape, output->dataType(), block.launchContext());

    // gate = X @ W_gate, up = X @ W_up (write into pre-allocated C-order buffers)
    MmulHelper::mmul(xFlat, wGate, gate, 1.0, 0.0);
    MmulHelper::mmul(xFlat, wUp,   up,   1.0, 0.0);

    if (needReshape)
        delete xFlat;

    // Compute silu(gate) * up into gate buffer (reusing gate as accumulator)
    // Step 1: sigmoidBuf = sigmoid(gate)
    auto sigmoidBuf = NDArrayFactory::create_('c', outFlatShape, output->dataType(), block.launchContext());
    gate->applyTransform(transform::Sigmoid, sigmoidBuf);
    // Step 2: gate *= sigmoid → gate = silu(gate)
    gate->applyPairwiseTransform(pairwise::Multiply, sigmoidBuf, gate);
    delete sigmoidBuf;
    // Step 3: gate *= up → gate = silu(gate) * up (final result in gate)
    gate->applyPairwiseTransform(pairwise::Multiply, up, gate);
    delete up;

    // Copy C-order flat result [M,N] to output [...,N].
    // Both are C-order with the same total elements, so assign handles it.
    output->assign(gate);
    delete gate;

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_gemm_swiglu) {
    auto xShape = inputShape->at(0);
    auto wGateShape = inputShape->at(1);
    auto xRank = shape::rank(xShape);
    auto N = shape::shapeOf(wGateShape)[1];
    auto dtype = ArrayOptions::dataType(xShape);

    // Output shape: leading dims from x + N from weight
    std::vector<sd::LongType> outShape;
    for (int d = 0; d < xRank - 1; d++)
        outShape.push_back(shape::shapeOf(xShape)[d]);
    outShape.push_back(N);

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', outShape));
}

DECLARE_TYPES(fused_gemm_swiglu) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(fused_gemm_swiglu_bp, 4, 3, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);       // [..., K]
    auto wGate = INPUT_VARIABLE(1);   // [K, N]
    auto wUp = INPUT_VARIABLE(2);     // [K, N]
    auto gradOut = INPUT_VARIABLE(3); // [..., N]
    auto dX = OUTPUT_VARIABLE(0);     // [..., K]
    auto dWGate = OUTPUT_VARIABLE(1); // [K, N]
    auto dWUp = OUTPUT_VARIABLE(2);   // [K, N]

    auto K = x->sizeAt(-1);
    auto N = wGate->sizeAt(1);

    // Flatten leading dims for rank > 2
    NDArray* xFlat = nullptr;
    NDArray* gradFlat = nullptr;
    sd::LongType M = 1;
    bool needReshape = x->rankOf() > 2;
    if (needReshape) {
        for (int d = 0; d < x->rankOf() - 1; d++)
            M *= x->sizeAt(d);
        std::vector<sd::LongType> flatX = {M, K};
        std::vector<sd::LongType> flatG = {M, N};
        xFlat = x->reshape('c', flatX);
        gradFlat = gradOut->reshape('c', flatG);
    } else {
        xFlat = x;
        gradFlat = gradOut;
        M = x->sizeAt(0);
    }

    // Recompute forward intermediates
    auto gate = MmulHelper::mmul(xFlat, wGate, nullptr, 1.0, 0.0);    // [M, N]
    auto up = MmulHelper::mmul(xFlat, wUp, nullptr, 1.0, 0.0);        // [M, N]

    // silu(gate) = gate * sigmoid(gate)
    NDArray* sigmoid = gate->transform(transform::Sigmoid);
    NDArray* siluGate = (*gate) * (*sigmoid);

    // d_up = gradOut * silu(gate)
    NDArray* dUpVal = (*gradFlat) * (*siluGate);

    // silu'(gate) = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
    NDArray* gateSig = (*gate) * (*sigmoid);          // gate * sigmoid
    NDArray* oneMinusSig = (*sigmoid) * (-1.0f);      // -sigmoid
    *oneMinusSig += 1.0f;                              // 1 - sigmoid
    NDArray* gsOms = (*gateSig) * (*oneMinusSig);     // gate * sigmoid * (1-sigmoid)
    delete gateSig;
    delete oneMinusSig;
    NDArray* siluDeriv = (*sigmoid) + (*gsOms);       // sigmoid + gate*sigmoid*(1-sigmoid)
    delete gsOms;

    // d_gate = gradOut * up * silu'(gate)
    NDArray* gradTimesUp = (*gradFlat) * (*up);
    NDArray* dGateVal = (*gradTimesUp) * (*siluDeriv);
    delete gradTimesUp;
    delete siluDeriv;
    delete sigmoid;
    delete siluGate;

    // d_W_gate = xFlat^T @ d_gate
    MmulHelper::matmul(xFlat, dGateVal, dWGate, true, false, 1.0, 0.0);

    // d_W_up = xFlat^T @ d_up
    MmulHelper::matmul(xFlat, dUpVal, dWUp, true, false, 1.0, 0.0);

    // d_x = d_gate @ W_gate^T + d_up @ W_up^T, then reshape back
    std::vector<sd::LongType> flatDxShape = {M, K};
    auto dXFlat = new NDArray('c', flatDxShape, x->dataType(), x->getContext());
    MmulHelper::matmul(dGateVal, const_cast<NDArray*>(wGate), dXFlat, false, true, 1.0, 0.0);
    auto dXUp = new NDArray('c', flatDxShape, x->dataType(), x->getContext());
    MmulHelper::matmul(dUpVal, const_cast<NDArray*>(wUp), dXUp, false, true, 1.0, 0.0);
    *dXFlat += *dXUp;

    if (needReshape) {
        auto dxShapePtr = dX->getShapeAsVector();
        auto reshaped = dXFlat->reshape('c', *dxShapePtr);
        dX->assign(reshaped);
        delete dxShapePtr;
    } else {
        dX->assign(dXFlat);
    }
    delete dXFlat;
    delete dXUp;

    delete dGateVal;
    delete dUpVal;
    delete gate;
    delete up;

    if (needReshape) {
        delete xFlat;
        delete gradFlat;
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_gemm_swiglu_bp) {
    auto xShape = inputShape->at(0);
    auto wGateShape = inputShape->at(1);
    auto wUpShape = inputShape->at(2);

    auto dXShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', shape::rank(xShape), shape::shapeOf(xShape));
    auto dWGateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(wGateShape), 'c', shape::rank(wGateShape), shape::shapeOf(wGateShape));
    auto dWUpShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(wUpShape), 'c', shape::rank(wUpShape), shape::shapeOf(wUpShape));

    return SHAPELIST(dXShape, dWGateShape, dWUpShape);
}

DECLARE_TYPES(fused_gemm_swiglu_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// rms_norm_linear - Fused RMSNorm + Linear: matmul(rms_norm(x, gamma, eps), W)
// Single-pass kernel computes both the normalization and linear projection,
// eliminating the intermediate normalized tensor from HBM.
// For now, implemented as rms_norm + matmul. The CUDA kernel with joint
// Σx² and Σx·W accumulation will be added as a platform-specific op.
#if NOT_EXCLUDED(OP_rms_norm_linear)
CUSTOM_OP_IMPL(rms_norm_linear, 3, 1, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);       // [..., K] input (rank >= 2)
    auto gamma = INPUT_VARIABLE(1);   // [K] scale
    auto w = INPUT_VARIABLE(2);       // [K, N] weight matrix
    auto output = OUTPUT_VARIABLE(0); // [..., N]

    float epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-6f;

    REQUIRE_TRUE(x->rankOf() >= 2, 0, "rms_norm_linear: input must be rank >= 2, got %d", x->rankOf());
    REQUIRE_TRUE(gamma->rankOf() == 1, 0, "rms_norm_linear: gamma must be rank 1, got %d", gamma->rankOf());
    REQUIRE_TRUE(w->rankOf() == 2, 0, "rms_norm_linear: weight must be rank 2, got %d", w->rankOf());

    const auto K = x->sizeAt(-1);
    REQUIRE_TRUE(gamma->lengthOf() == K, 0,
                 "rms_norm_linear: gamma length must match input's last dimension, got %lld vs %lld",
                 (long long)gamma->lengthOf(), (long long)K);
    REQUIRE_TRUE(w->sizeAt(0) == K, 0,
                 "rms_norm_linear: weight rows must match input's last dimension, got %lld vs %lld",
                 (long long)w->sizeAt(0), (long long)K);

    // Helpers preserve the input/output dtype while reading floating gamma and
    // weight arrays in their native dtypes, so no op-level casts are needed.

    // For rank-3+ inputs, flatten leading dims to rank-2 for the helper
    if (x->rankOf() > 2) {
        auto K = x->sizeAt(-1);
        auto M = x->lengthOf() / K;
        auto N = w->sizeAt(1);
        std::vector<sd::LongType> xShape2d = {M, K};
        std::vector<sd::LongType> outShape2d = {M, N};
        auto x2d = x->reshape('c', xShape2d, false);
        auto out2d = output->reshape('c', outShape2d, false);
        const bool directWrite = out2d->dataBuffer() == output->dataBuffer();
        helpers::rmsNormLinear(block.launchContext(), x2d, gamma, w, out2d, epsilon);
        if (!directWrite) {
            auto outShape = output->getShapeAsVector();
            auto reshaped = out2d->reshape(output->ordering(), *outShape, false);
            output->assign(reshaped);
            delete reshaped;
            delete outShape;
        }
        delete x2d;
        delete out2d;
    } else {
        helpers::rmsNormLinear(block.launchContext(), x, gamma, w, output, epsilon);
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(rms_norm_linear) {
    auto xShape = inputShape->at(0);
    auto wShape = inputShape->at(2);

    auto xRank = shape::rank(xShape);
    auto N = shape::shapeOf(wShape)[1];
    auto dtype = ArrayOptions::dataType(xShape);

    // Preserve all leading dims, replace last dim with N
    // e.g. [B, S, K] @ [K, N] -> [B, S, N]
    //      [M, K] @ [K, N] -> [M, N]
    std::vector<sd::LongType> outShape;
    for (int i = 0; i < xRank - 1; i++) {
        outShape.push_back(shape::shapeOf(xShape)[i]);
    }
    outShape.push_back(N);

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', outShape));
}

DECLARE_TYPES(rms_norm_linear) {
  getOpDescriptor()->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(rms_norm_linear_bp, 4, 3, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);       // [M, K]
    auto gamma = INPUT_VARIABLE(1);   // [K]
    auto w = INPUT_VARIABLE(2);       // [K, N]
    auto gradOut = INPUT_VARIABLE(3); // [M, N]
    auto dX = OUTPUT_VARIABLE(0);     // [M, K]
    auto dGamma = OUTPUT_VARIABLE(1); // [K]
    auto dW = OUTPUT_VARIABLE(2);     // [K, N]

    float epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-6f;

    // Step 1: Recompute normalized = rms_norm(x, gamma, eps)
    auto K = x->sizeAt(-1);
    std::vector<LongType> lastDim = {x->rankOf() - 1};

    NDArray* xSquared = (*x) * (*x);
    NDArray* meanSquared = xSquared->reduceAlongDimension(reduce::Mean, &lastDim, true);
    delete xSquared;
    NDArray* meanPlusEps = (*meanSquared) + epsilon;
    delete meanSquared;
    NDArray* sqrtVal = meanPlusEps->transform(transform::Sqrt);
    delete meanPlusEps;
    NDArray* rsqrtVal = sqrtVal->transform(transform::Reciprocal);
    delete sqrtVal;
    NDArray* normalized = (*x) * (*rsqrtVal);  // x * rsqrt(mean(x^2) + eps)
    NDArray* scaled = (*normalized) * (*gamma); // normalized * gamma

    // Step 2: mmul backward
    // d_normalized_scaled = gradOut @ W^T
    auto dScaled = new NDArray(x->shapeInfo(), false, x->getContext());
    MmulHelper::matmul(gradOut, const_cast<NDArray*>(w), dScaled, false, true, 1.0, 0.0);

    // d_W = scaled^T @ gradOut
    MmulHelper::matmul(scaled, gradOut, dW, true, false, 1.0, 0.0);
    delete scaled;

    // Step 3: rms_norm backward (gamma scaling)
    // d_normalized = d_scaled / gamma (element-wise, broadcast along last dim)
    // But actually d_scaled = d(normalized * gamma) so d_normalized = d_scaled * gamma
    // and d_gamma = sum(d_scaled * normalized, axis=0..rank-2)
    NDArray* dNormalized = (*dScaled) * (*gamma);
    // d_gamma = sum over batch dims of (d_scaled * normalized)
    NDArray* dGammaFull = (*dScaled) * (*normalized);
    delete dScaled;

    if (x->rankOf() > 1) {
        std::vector<LongType> batchDims;
        for (int d = 0; d < x->rankOf() - 1; d++) batchDims.push_back(d);
        NDArray* dGammaReduced = dGammaFull->reduceAlongDimension(reduce::Sum, &batchDims, false);
        dGamma->assign(dGammaReduced);
        delete dGammaReduced;
    } else {
        dGamma->assign(dGammaFull);
    }
    delete dGammaFull;

    // Step 4: rms_norm backward (input gradient)
    // normalized = x * rsqrt, so d_x involves the rsqrt derivative
    // d_x = rsqrt * (d_normalized - normalized * mean(d_normalized * normalized))
    NDArray* dnTimesNorm = (*dNormalized) * (*normalized);
    NDArray* meanDnNorm = dnTimesNorm->reduceAlongDimension(reduce::Mean, &lastDim, true);
    delete dnTimesNorm;

    NDArray* correction = (*normalized) * (*meanDnNorm);
    delete meanDnNorm;
    NDArray* adjusted = (*dNormalized) - (*correction);
    delete correction;
    delete dNormalized;

    NDArray* dXCalc = (*adjusted) * (*rsqrtVal);
    delete adjusted;
    delete rsqrtVal;
    delete normalized;

    dX->assign(dXCalc);
    delete dXCalc;

    return Status::OK;
}

DECLARE_SHAPE_FN(rms_norm_linear_bp) {
    auto xShape = inputShape->at(0);
    auto gammaShape = inputShape->at(1);
    auto wShape = inputShape->at(2);

    auto dXShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', shape::rank(xShape), shape::shapeOf(xShape));
    auto dGammaShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(gammaShape), 'c', shape::rank(gammaShape), shape::shapeOf(gammaShape));
    auto dWShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(wShape), 'c', shape::rank(wShape), shape::shapeOf(wShape));

    return SHAPELIST(dXShape, dGammaShape, dWShape);
}

DECLARE_TYPES(rms_norm_linear_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// mean_square - Mean of squared values
#if NOT_EXCLUDED(OP_mean_square)
CUSTOM_OP_IMPL(mean_square, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    bool keepDims = block.getIArguments()->size() > 0 ? INT_ARG(0) != 0 : true;

    // mean(x * x) along last dimension
    NDArray* squared = (*input) * (*input);
    std::vector<LongType> axis = {input->rankOf() - 1};
    NDArray* meanSquared = squared->reduceAlongDimension(reduce::Mean, &axis, keepDims);
    delete squared;

    output->assign(meanSquared);
    delete meanSquared;

    return Status::OK;
}

DECLARE_SHAPE_FN(mean_square) {
    auto inShape = inputShape->at(0);
    auto rank = shape::rank(inShape);
    auto dtype = ArrayOptions::dataType(inShape);

    bool keepDims = block.getIArguments()->size() > 0 ? INT_ARG(0) != 0 : true;

    if (keepDims) {
        // Same shape but last dimension = 1
        std::vector<LongType> outShape;
        for (int i = 0; i < rank - 1; i++) {
            outShape.push_back(shape::sizeAt(inShape, static_cast<LongType>(i)));
        }
        outShape.push_back(1);
        return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', outShape));
    } else {
        // Reduced shape without last dimension
        std::vector<LongType> outShape;
        for (int i = 0; i < rank - 1; i++) {
            outShape.push_back(shape::sizeAt(inShape, static_cast<LongType>(i)));
        }
        if (outShape.empty()) {
            outShape.push_back(1);  // Scalar case
        }
        return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', outShape));
    }
}

DECLARE_TYPES(mean_square) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(mean_square_bp, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gradOut = INPUT_VARIABLE(1);
    auto gradIn = OUTPUT_VARIABLE(0);

    bool keepDims = block.getIArguments()->size() > 0 ? INT_ARG(0) != 0 : true;

    std::vector<LongType> axis = {input->rankOf() - 1};
    auto n = input->sizeAt(axis[0]);

    // d/dx[mean(x^2)] = 2*x / n
    NDArray* grad = (*input) * (2.0f / static_cast<float>(n));

    // Broadcast gradOut to match input shape and multiply with grad
    NDArray* result = (*grad) * (*gradOut);
    gradIn->assign(result);
    delete result;
    delete grad;

    return Status::OK;
}

DECLARE_SHAPE_FN(mean_square_bp) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(mean_square_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// column_parallel_linear - Column-Parallel Linear for Tensor Parallelism
#if NOT_EXCLUDED(OP_column_parallel_linear)
CUSTOM_OP_IMPL(column_parallel_linear, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);       // [batch, in_features]
    auto weightShard = INPUT_VARIABLE(1); // [in_features, out_features/tp_size]
    auto output = OUTPUT_VARIABLE(0);

    NDArray* biasShard = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

    int tpSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
    int tpRank = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int gatherOutput = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;

    // Local matmul: output_shard = input @ weightShard
    MmulHelper::mmul(input, weightShard, output);

    // Add bias if present
    if (biasShard != nullptr) {
        *output += *biasShard;
    }

    // AllGather is handled at the Java level via NcclCommunicator.
    // When tpSize=1, this op is just a standard linear layer.

    return Status::OK;
}

DECLARE_SHAPE_FN(column_parallel_linear) {
    auto inShape = inputShape->at(0);
    auto wShape = inputShape->at(1);

    auto batchDim = shape::sizeAt(inShape, 0);
    auto outDim = shape::sizeAt(wShape, -1);

    auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inShape), 'c', {batchDim, outDim});

    return SHAPELIST(outShape);
}

DECLARE_TYPES(column_parallel_linear) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// row_parallel_linear - Row-Parallel Linear for Tensor Parallelism
#if NOT_EXCLUDED(OP_row_parallel_linear)
CUSTOM_OP_IMPL(row_parallel_linear, 2, 1, false, 0, 0) {
    auto inputShard = INPUT_VARIABLE(0);  // [batch, in_features/tp_size]
    auto weightShard = INPUT_VARIABLE(1); // [in_features/tp_size, out_features]
    auto output = OUTPUT_VARIABLE(0);

    NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

    int tpSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
    int tpRank = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int reduceOutput = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;

    // Local matmul: partial = inputShard @ weightShard
    MmulHelper::mmul(inputShard, weightShard, output);

    // AllReduce is handled at the Java level via NcclCommunicator.
    // Bias is added AFTER AllReduce at Java level (only on rank 0 or after reduce).
    // When tpSize=1, add bias here directly.
    if (bias != nullptr && tpSize <= 1) {
        *output += *bias;
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(row_parallel_linear) {
    auto inShape = inputShape->at(0);
    auto wShape = inputShape->at(1);

    auto batchDim = shape::sizeAt(inShape, 0);
    auto outDim = shape::sizeAt(wShape, -1);

    auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inShape), 'c', {batchDim, outDim});

    return SHAPELIST(outShape);
}

DECLARE_TYPES(row_parallel_linear) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// kv_cache_quantize - KV Cache Quantization
#if NOT_EXCLUDED(OP_kv_cache_quantize)
CUSTOM_OP_IMPL(kv_cache_quantize, 1, 2, false, 0, 1) {
    auto input = INPUT_VARIABLE(0);
    auto quantized = OUTPUT_VARIABLE(0);
    auto scales = OUTPUT_VARIABLE(1);

    int quantFormat = INT_ARG(0);
    // ADR 0107 V2 ROW-INLINE (INT8 only): when the 2nd I-arg is 1, OUTPUT 0 is a row-inline INT8
    // tensor of the input shape with the last dimension extended by 4 — each row holds rowLen
    // int8 values followed by that row's float32 scale. The scale rides INSIDE the logical
    // tensor, so any DSP staging/copy preserves it. OUTPUT 1 is an unused dummy scalar.
    // helpers::kvCacheQuantize writes the in-row scales when passed a null scales array.
    const bool inlineScale = (block.numI() > 1 && INT_ARG(1) != 0);

    if (inlineScale) {
        helpers::kvCacheQuantize(input, quantized, /*scales=*/nullptr, quantFormat, block.launchContext());
    } else {
        helpers::kvCacheQuantize(input, quantized, scales, quantFormat, block.launchContext());
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(kv_cache_quantize) {
    auto inShape = inputShape->at(0);
    auto rank = shape::rank(inShape);
    const bool inlineScale = (block.numI() > 1 && INT_ARG(1) != 0);

    // Row layout: rows span dims[0..rank-2], rowLen = dims[rank-1]; one float scale per row.
    const LongType rowLen = (rank >= 1) ? shape::sizeAt(inShape, rank - 1) : 1;

    if (inlineScale) {
        // Output 0: ROW-INLINE INT8 tensor — input shape with the last dimension extended by 4
        // (each row = rowLen int8 values ++ that row's float32 scale). Rank-preserved so the
        // tensor's logical bytes cover the scales and survive any staging/copy.
        std::vector<LongType> rowInlineShape;
        for (int i = 0; i < rank - 1; ++i) rowInlineShape.push_back(shape::sizeAt(inShape, i));
        rowInlineShape.push_back(rowLen + 4);
        auto rowInlineInfo = ConstantShapeHelper::getInstance().createShapeInfo(
            DataType::INT8, 'c', rowInlineShape);
        // Output 1: unused dummy scalar (scale rides inline in output 0).
        auto dummyInfo = ConstantShapeHelper::getInstance().createShapeInfo(
            DataType::FLOAT32, 'c', std::vector<LongType>{1});
        return new ShapeList(std::vector<LongType*>{rowInlineInfo, dummyInfo});
    }

    // Output 0: quantized data — same shape as input, INT8 dtype
    auto quantShape = ConstantShapeHelper::getInstance().createShapeInfo(
        DataType::INT8, shape::order(inShape), static_cast<int>(shape::rank(inShape)), shape::shapeOf(inShape), static_cast<LongType>(0));

    // Output 1: scales — input shape with last dimension removed (one scale per row)
    std::vector<LongType> scaleShapeVec;
    for (int i = 0; i < rank - 1; ++i) {
        scaleShapeVec.push_back(shape::sizeAt(inShape, i));
    }
    if (scaleShapeVec.empty()) {
        scaleShapeVec.push_back(1);
    }

    auto scaleShape = ConstantShapeHelper::getInstance().createShapeInfo(
        DataType::FLOAT32, 'c', scaleShapeVec);

    return new ShapeList(std::vector<LongType*>{quantShape, scaleShape});
}

DECLARE_TYPES(kv_cache_quantize) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes(0, {DataType::INT8});
    getOpDescriptor()->setAllowedOutputTypes(1, {DataType::FLOAT32});
    getOpDescriptor()->addTraits(OP_TRAIT_CONSTANT_GENERATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_VALUE_DEPENDENT_SHAPE);
}
#endif

//////////////////////////////////////////////////////////////////////////
// kv_cache_dequantize - KV Cache Dequantization
#if NOT_EXCLUDED(OP_kv_cache_dequantize)
CUSTOM_OP_IMPL(kv_cache_dequantize, 2, 1, false, 0, 1) {
    auto quantized = INPUT_VARIABLE(0);
    auto scales = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    int quantFormat = INT_ARG(0);

    helpers::kvCacheDequantize(quantized, scales, output, quantFormat, block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(kv_cache_dequantize) {
    auto quantShape = inputShape->at(0);

    // Output: same shape as quantized input, FLOAT32 dtype
    auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
        DataType::FLOAT32, shape::order(quantShape), static_cast<int>(shape::rank(quantShape)), shape::shapeOf(quantShape), static_cast<LongType>(0));

    return SHAPELIST(outShape);
}

DECLARE_TYPES(kv_cache_dequantize) {
    getOpDescriptor()->setAllowedInputTypes(0, {DataType::INT8, DataType::UINT8});
    getOpDescriptor()->setAllowedInputTypes(1, {DataType::FLOAT32});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}
#endif

//////////////////////////////////////////////////////////////////////////
// ggml_dequantize - Dequantize raw GGML quantized bytes to target float type
#if NOT_EXCLUDED(OP_ggml_dequantize)
CUSTOM_OP_IMPL(ggml_dequantize, 1, 1, false, 0, 1) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int quantType = INT_ARG(0);

    helpers::ggmlDequantize(block.launchContext(), input, output, quantType);

    return Status::OK;
}

DECLARE_SHAPE_FN(ggml_dequantize) {
    // iArgs[0] = quant type, iArgs[1] = output dtype, iArgs[2..N] = output shape dimensions
    auto iArgs = block.getIArguments();
    REQUIRE_TRUE(iArgs->size() >= 3, 0, "ggml_dequantize: need at least 3 iArgs (quantType, outputDtype, 1+ shape dims)");

    int outputDtypeArg = iArgs->at(1);
    DataType outputDtype;
    switch (outputDtypeArg) {
        case 0: outputDtype = DataType::FLOAT32; break;
        case 1: outputDtype = DataType::HALF; break;
        case 2: outputDtype = DataType::BFLOAT16; break;
        default: outputDtype = DataType::FLOAT32; break;
    }

    std::vector<LongType> outShape;
    for (size_t i = 2; i < iArgs->size(); i++) {
        outShape.push_back(iArgs->at(i));
    }

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        outputDtype, 'c', outShape));
}

DECLARE_TYPES(ggml_dequantize) {
    getOpDescriptor()->setAllowedInputTypes({ALL_INTS, ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({DataType::FLOAT32, DataType::HALF, DataType::BFLOAT16});
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

// llama.cpp-compat name (same contract: iArg 0 = GGML quant type of the input bytes)
DECLARE_SYN(dequantize, ggml_dequantize);
#endif

}  // namespace ops
}  // namespace sd

#endif
// NOTE: fused_gelu, fused_layer_norm, fused_rope, fused_bias_dropout_residual,
// and fused_rms_norm_swiglu are implemented in fused_llm_ops.cpp which calls
// the platform-specific helpers.
