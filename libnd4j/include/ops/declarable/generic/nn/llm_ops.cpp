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
#include <helpers/ShapeUtils.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/kv_cache_quantize.h>
#include <ops/declarable/helpers/ggml_dequantize.h>
#include <cmath>

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

CUSTOM_OP_IMPL(rms_norm_bp, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gradOut = INPUT_VARIABLE(1);
    auto gradIn = OUTPUT_VARIABLE(0);

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;
    std::vector<LongType> axis = {input->rankOf() - 1};
    auto n = input->sizeAt(axis[0]);

    // Forward pass values
    NDArray* squared = (*input) * (*input);
    NDArray* meanSquared = squared->reduceAlongDimension(reduce::Mean, &axis, true);
    delete squared;

    NDArray* meanPlusEps = (*meanSquared) + eps;
    delete meanSquared;
    NDArray* rsqrt = meanPlusEps->transform(transform::RSqrt);
    delete meanPlusEps;

    // Gradient computation
    NDArray* gradNorm = (*gradOut) * (*rsqrt);
    NDArray* inputGrad = (*input) * (*gradOut);
    NDArray* dotProduct = inputGrad->reduceAlongDimension(reduce::Sum, &axis, true);
    delete inputGrad;

    NDArray* rsqrt2 = (*rsqrt) * (*rsqrt);
    NDArray* rsqrt3 = (*rsqrt2) * (*rsqrt);
    delete rsqrt2;
    delete rsqrt;

    NDArray* gradMean = (*dotProduct) * (*rsqrt3);
    delete dotProduct;
    delete rsqrt3;

    NDArray* gradMeanScaled = (*gradMean) / static_cast<float>(n);
    delete gradMean;

    NDArray* inputScaled = (*input) * (*gradMeanScaled);
    delete gradMeanScaled;

    NDArray* result = (*gradNorm) - (*inputScaled);
    delete gradNorm;
    delete inputScaled;

    gradIn->assign(result);
    delete result;

    return Status::OK;
}

DECLARE_SHAPE_FN(rms_norm_bp) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
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

    int mode = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int nPast = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int nDims = block.getIArguments()->size() > 2 ? INT_ARG(2) : static_cast<int>(input->sizeAt(-1));
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

    auto batch = input->sizeAt(0);
    auto seqLen = input->sizeAt(1);
    auto numHeads = input->sizeAt(2);
    auto headDim = input->sizeAt(3);

    output->assign(input);
    auto outputBuf = output->bufferAsT<float>();

    // Apply rotary embeddings
    for (LongType b = 0; b < batch; ++b) {
        for (LongType s = 0; s < seqLen; ++s) {
            LongType pos = nPast + s;
            for (LongType h = 0; h < numHeads; ++h) {
                for (int i = 0; i < nDims / 2; ++i) {
                    float theta = static_cast<float>(pos) * freqScale /
                                  std::pow(freqBase, (2.0f * i) / nDims);
                    float cosTheta = std::cos(theta);
                    float sinTheta = std::sin(theta);

                    LongType idx1 = ((b * seqLen + s) * numHeads + h) * headDim + i;
                    LongType idx2 = ((b * seqLen + s) * numHeads + h) * headDim + i + nDims / 2;

                    float x1 = outputBuf[idx1];
                    float x2 = outputBuf[idx2];

                    outputBuf[idx1] = x1 * cosTheta - x2 * sinTheta;
                    outputBuf[idx2] = x1 * sinTheta + x2 * cosTheta;
                }
            }
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(rope) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(rope) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
}

CUSTOM_OP_IMPL(rope_bp, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gradOut = INPUT_VARIABLE(1);
    auto gradIn = OUTPUT_VARIABLE(0);

    int nPast = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int nDims = block.getIArguments()->size() > 2 ? INT_ARG(2) : static_cast<int>(input->sizeAt(-1));
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

    auto batch = input->sizeAt(0);
    auto seqLen = input->sizeAt(1);
    auto numHeads = input->sizeAt(2);
    auto headDim = input->sizeAt(3);

    gradIn->assign(gradOut);
    auto gradBuf = gradIn->bufferAsT<float>();

    // Backward pass: apply inverse rotation
    for (LongType b = 0; b < batch; ++b) {
        for (LongType s = 0; s < seqLen; ++s) {
            LongType pos = nPast + s;
            for (LongType h = 0; h < numHeads; ++h) {
                for (int i = 0; i < nDims / 2; ++i) {
                    float theta = static_cast<float>(pos) * freqScale /
                                  std::pow(freqBase, (2.0f * i) / nDims);
                    float cosTheta = std::cos(theta);
                    float sinTheta = std::sin(theta);

                    LongType idx1 = ((b * seqLen + s) * numHeads + h) * headDim + i;
                    LongType idx2 = ((b * seqLen + s) * numHeads + h) * headDim + i + nDims / 2;

                    float g1 = gradBuf[idx1];
                    float g2 = gradBuf[idx2];

                    // Inverse rotation
                    gradBuf[idx1] = g1 * cosTheta + g2 * sinTheta;
                    gradBuf[idx2] = -g1 * sinTheta + g2 * cosTheta;
                }
            }
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(rope_bp) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

DECLARE_TYPES(rope_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}
#endif

//////////////////////////////////////////////////////////////////////////
// silu - SiLU/Swish Activation
#if NOT_EXCLUDED(OP_silu)
CONFIGURABLE_OP_IMPL(silu, 1, 1, true, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // silu(x) = x * sigmoid(x)
    NDArray* sigmoid = input->transform(transform::Sigmoid);
    NDArray* result = (*input) * (*sigmoid);
    output->assign(result);
    delete sigmoid;
    delete result;

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
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_ACTIVATION | OP_TRAIT_BACKWARD);
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
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}
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
    auto computedOutput = NDArrayFactory::create_<float>('c', *queryShape);
    delete queryShape;
    auto seqLen = query->sizeAt(1);
    auto numHeads = query->sizeAt(2);
    auto batch = query->sizeAt(0);
    std::vector<sd::LongType> lseShape = {batch, numHeads, seqLen};
    auto computedLse = NDArrayFactory::create_<float>('c', lseShape);

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
        computedOutput = NDArrayFactory::create_<float>('c', *queryShapeVec);
        delete queryShapeVec;
        auto seqLen = query->sizeAt(1);
        auto numHeads = query->sizeAt(2);
        auto batch = query->sizeAt(0);
        std::vector<sd::LongType> lseShapeVec = {batch, numHeads, seqLen};
        computedLse = NDArrayFactory::create_<float>('c', lseShapeVec);

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

    // Update with new keys/values at position using raw buffer copy
    auto newSeqLen = newKeys->sizeAt(1);
    auto batch = newKeys->sizeAt(0);
    auto numHeads = newKeys->rankOf() > 2 ? newKeys->sizeAt(2) : 1;
    auto headDim = newKeys->rankOf() > 3 ? newKeys->sizeAt(3) : newKeys->sizeAt(-1);
    auto cacheSeqLen = keyCache->sizeAt(1);

    auto newKeyBuf = newKeys->bufferAsT<float>();
    auto newValueBuf = newValues->bufferAsT<float>();
    auto outKeyBuf = outputKeyCache->bufferAsT<float>();
    auto outValueBuf = outputValueCache->bufferAsT<float>();

    // Copy new keys/values into cache at the specified position
    for (LongType b = 0; b < batch; ++b) {
        for (LongType i = 0; i < newSeqLen; ++i) {
            for (LongType h = 0; h < numHeads; ++h) {
                LongType srcBase = ((b * newSeqLen + i) * numHeads + h) * headDim;
                LongType dstBase = ((b * cacheSeqLen + startPos + i) * numHeads + h) * headDim;
                PRAGMA_OMP_SIMD
                for (LongType d = 0; d < headDim; ++d) {
                    outKeyBuf[dstBase + d] = newKeyBuf[srcBase + d];
                    outValueBuf[dstBase + d] = newValueBuf[srcBase + d];
                }
            }
        }
    }

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

    auto outputBuf = output->bufferAsT<float>();

    // Compute ALiBi slopes
    std::vector<float> slopes(numHeads);
    float base = std::pow(2.0f, -8.0f / numHeads);
    for (int h = 0; h < numHeads; ++h) {
        slopes[h] = std::pow(base, h + 1);
    }

    // Apply ALiBi bias
    for (LongType b = 0; b < batch; ++b) {
        for (int h = 0; h < numHeads; ++h) {
            for (LongType sq = 0; sq < seqLen; ++sq) {
                for (LongType sk = 0; sk < kvLen; ++sk) {
                    LongType idx = ((b * numHeads + h) * seqLen + sq) * kvLen + sk;
                    // ALiBi: subtract slope * |query_pos - key_pos|
                    float bias = -slopes[h] * std::abs(static_cast<float>(sq) - static_cast<float>(sk));
                    outputBuf[idx] += bias;
                }
            }
        }
    }

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

    // swish_mul(x, y) = silu(x) * y = x * sigmoid(x) * y
    NDArray* sigmoid = x->transform(transform::Sigmoid);
    NDArray* swish = (*x) * (*sigmoid);
    delete sigmoid;

    NDArray* result = (*swish) * (*y);
    delete swish;

    output->assign(result);
    delete result;

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
    auto x = INPUT_VARIABLE(0);       // [M, K] input activations
    auto wGate = INPUT_VARIABLE(1);   // [K, N] gate weight matrix
    auto wUp = INPUT_VARIABLE(2);     // [K, N] up weight matrix
    auto output = OUTPUT_VARIABLE(0); // [M, N]

    REQUIRE_TRUE(x->rankOf() >= 2, 0, "fused_gemm_swiglu: input must be rank >= 2, got %d", x->rankOf());
    REQUIRE_TRUE(wGate->rankOf() == 2, 0, "fused_gemm_swiglu: wGate must be rank 2, got %d", wGate->rankOf());
    REQUIRE_TRUE(wUp->rankOf() == 2, 0, "fused_gemm_swiglu: wUp must be rank 2, got %d", wUp->rankOf());

    // gate = X @ W_gate
    auto gate = MmulHelper::mmul(x, wGate, nullptr, 1.0, 0.0);

    // up = X @ W_up
    auto up = MmulHelper::mmul(x, wUp, nullptr, 1.0, 0.0);

    // out = silu(gate) * up = gate * sigmoid(gate) * up
    NDArray* sigmoid = gate->transform(transform::Sigmoid);
    NDArray* silu = (*gate) * (*sigmoid);
    delete sigmoid;
    delete gate;

    NDArray* result = (*silu) * (*up);
    delete silu;
    delete up;

    output->assign(result);
    delete result;

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_gemm_swiglu) {
    auto xShape = inputShape->at(0);
    auto wGateShape = inputShape->at(1);

    auto M = shape::shapeOf(xShape)[0];
    auto N = shape::shapeOf(wGateShape)[1];
    auto dtype = ArrayOptions::dataType(xShape);

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {M, N}));
}

DECLARE_TYPES(fused_gemm_swiglu) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

CUSTOM_OP_IMPL(fused_gemm_swiglu_bp, 4, 3, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);       // [M, K]
    auto wGate = INPUT_VARIABLE(1);   // [K, N]
    auto wUp = INPUT_VARIABLE(2);     // [K, N]
    auto gradOut = INPUT_VARIABLE(3); // [M, N]
    auto dX = OUTPUT_VARIABLE(0);     // [M, K]
    auto dWGate = OUTPUT_VARIABLE(1); // [K, N]
    auto dWUp = OUTPUT_VARIABLE(2);   // [K, N]

    // Recompute forward intermediates
    auto gate = MmulHelper::mmul(x, wGate, nullptr, 1.0, 0.0);    // [M, N]
    auto up = MmulHelper::mmul(x, wUp, nullptr, 1.0, 0.0);        // [M, N]

    // silu(gate) = gate * sigmoid(gate)
    NDArray* sigmoid = gate->transform(transform::Sigmoid);
    NDArray* siluGate = (*gate) * (*sigmoid);

    // d_up = gradOut * silu(gate)
    NDArray* dUpVal = (*gradOut) * (*siluGate);

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
    NDArray* gradTimesUp = (*gradOut) * (*up);
    NDArray* dGateVal = (*gradTimesUp) * (*siluDeriv);
    delete gradTimesUp;
    delete siluDeriv;
    delete sigmoid;
    delete siluGate;

    // d_W_gate = x^T @ d_gate
    MmulHelper::matmul(x, dGateVal, dWGate, true, false, 1.0, 0.0);

    // d_W_up = x^T @ d_up
    MmulHelper::matmul(x, dUpVal, dWUp, true, false, 1.0, 0.0);

    // d_x = d_gate @ W_gate^T + d_up @ W_up^T
    auto dXGate = new NDArray(dX->shapeInfo(), false, x->getContext());
    MmulHelper::matmul(dGateVal, const_cast<NDArray*>(wGate), dXGate, false, true, 1.0, 0.0);
    auto dXUp = new NDArray(dX->shapeInfo(), false, x->getContext());
    MmulHelper::matmul(dUpVal, const_cast<NDArray*>(wUp), dXUp, false, true, 1.0, 0.0);
    *dXGate += *dXUp;
    dX->assign(dXGate);
    delete dXGate;
    delete dXUp;

    delete dGateVal;
    delete dUpVal;
    delete gate;
    delete up;

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_gemm_swiglu_bp) {
    auto xShape = inputShape->at(0);
    auto wGateShape = inputShape->at(1);
    auto wUpShape = inputShape->at(2);

    auto dXShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', shape::shapeOf(xShape), shape::rank(xShape));
    auto dWGateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(wGateShape), 'c', shape::shapeOf(wGateShape), shape::rank(wGateShape));
    auto dWUpShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(wUpShape), 'c', shape::shapeOf(wUpShape), shape::rank(wUpShape));

    return SHAPELIST(dXShape, dWGateShape, dWUpShape);
}

DECLARE_TYPES(fused_gemm_swiglu_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
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
    auto x = INPUT_VARIABLE(0);       // [M, K] input
    auto gamma = INPUT_VARIABLE(1);   // [K] scale
    auto w = INPUT_VARIABLE(2);       // [K, N] weight matrix
    auto output = OUTPUT_VARIABLE(0); // [M, N]

    float epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-6f;

    REQUIRE_TRUE(x->rankOf() >= 2, 0, "rms_norm_linear: input must be rank >= 2, got %d", x->rankOf());
    REQUIRE_TRUE(w->rankOf() == 2, 0, "rms_norm_linear: weight must be rank 2, got %d", w->rankOf());

    helpers::rmsNormLinear(block.launchContext(), x, gamma, w, output, epsilon);

    return Status::OK;
}

DECLARE_SHAPE_FN(rms_norm_linear) {
    auto xShape = inputShape->at(0);
    auto wShape = inputShape->at(2);

    auto M = shape::shapeOf(xShape)[0];
    auto N = shape::shapeOf(wShape)[1];
    auto dtype = ArrayOptions::dataType(xShape);

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {M, N}));
}

DECLARE_TYPES(rms_norm_linear) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING);
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
        ArrayOptions::dataType(xShape), 'c', shape::shapeOf(xShape), shape::rank(xShape));
    auto dGammaShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(gammaShape), 'c', shape::shapeOf(gammaShape), shape::rank(gammaShape));
    auto dWShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(wShape), 'c', shape::shapeOf(wShape), shape::rank(wShape));

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
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
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
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    getOpDescriptor()->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
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

#if defined(SD_CUDA)
    helpers::kvCacheQuantizeCuda(input, quantized, scales, quantFormat, block.launchContext());
#else
    helpers::kvCacheQuantizeCpu(input, quantized, scales, quantFormat);
#endif

    return Status::OK;
}

DECLARE_SHAPE_FN(kv_cache_quantize) {
    auto inShape = inputShape->at(0);
    auto rank = shape::rank(inShape);

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

#if defined(SD_CUDA)
    helpers::kvCacheDequantizeCuda(quantized, scales, output, quantFormat, block.launchContext());
#else
    helpers::kvCacheDequantizeCpu(quantized, scales, output, quantFormat);
#endif

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
#endif

}  // namespace ops
}  // namespace sd

#endif
// NOTE: fused_gelu, fused_layer_norm, fused_rope, fused_bias_dropout_residual,
// and fused_rms_norm_swiglu are implemented in fused_llm_ops.cpp which calls
// the platform-specific helpers.
