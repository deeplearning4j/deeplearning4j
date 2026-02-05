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
    NOT_EXCLUDED(OP_swish_mul) || NOT_EXCLUDED(OP_mean_square)

#include <ops/declarable/headers/llm.h>
#include <helpers/MmulHelper.h>
#include <helpers/FlashAttentionHelper.h>
#include <math/templatemath.h>
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

    // RMS = sqrt(mean(x^2))
    NDArray* squared = (*input) * (*input);
    std::vector<LongType> axis = {input->rankOf() - 1};
    NDArray* meanSquared = squared->reduceAlongDimension(reduce::Mean, &axis, true);
    delete squared;

    // rsqrt = 1 / sqrt(mean + eps)
    NDArray* meanPlusEps = (*meanSquared) + eps;
    delete meanSquared;
    NDArray* rsqrt = meanPlusEps->transform(transform::RSqrt);
    delete meanPlusEps;

    // output = input * rsqrt
    NDArray* result = (*input) * (*rsqrt);
    output->assign(result);
    delete result;
    delete rsqrt;

    // Apply gamma if provided
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

}  // namespace ops
}  // namespace sd

#endif
// NOTE: fused_gelu, fused_layer_norm, fused_rope, fused_bias_dropout_residual,
// and fused_rms_norm_swiglu are implemented in fused_llm_ops.cpp which calls
// the platform-specific helpers.
