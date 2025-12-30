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
#if NOT_EXCLUDED(OP_rms_norm) || NOT_EXCLUDED(OP_rope) || NOT_EXCLUDED(OP_silu) || \
    NOT_EXCLUDED(OP_quantized_matmul) || NOT_EXCLUDED(OP_grouped_query_attention) || \
    NOT_EXCLUDED(OP_flash_attention) || NOT_EXCLUDED(OP_kv_cache_update) || \
    NOT_EXCLUDED(OP_apply_alibi) || NOT_EXCLUDED(OP_sliding_window_attention)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <helpers/MmulHelper.h>
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
    auto squared = *input * *input;
    int axis = input->rankOf() - 1;
    auto meanSquared = squared.reduceAlongDimension(reduce::Mean, {axis}, true);

    // rsqrt = 1 / sqrt(mean + eps)
    auto rsqrt = (meanSquared + eps).transform(transform::RSqrt);

    // output = input * rsqrt
    output->assign(*input * rsqrt);

    // Apply gamma if provided
    if (gamma != nullptr) {
        output->applyBroadcast(broadcast::Multiply, {axis}, *gamma, *output);
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(rms_norm) {
    auto inputShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape), shape::order(inputShape),
        shape::rank(inputShape), shape::shapeOf(inputShape)));
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
    int axis = input->rankOf() - 1;
    auto n = input->sizeAt(axis);

    // Forward pass values
    auto squared = *input * *input;
    auto meanSquared = squared.reduceAlongDimension(reduce::Mean, {axis}, true);
    auto rsqrt = (meanSquared + eps).transform(transform::RSqrt);

    // Gradient computation
    auto gradNorm = *gradOut * rsqrt;
    auto dotProduct = (*input * *gradOut).reduceAlongDimension(reduce::Sum, {axis}, true);
    auto gradMean = dotProduct * rsqrt * rsqrt * rsqrt / static_cast<float>(n);

    gradIn->assign(gradNorm - *input * gradMean);

    return Status::OK;
}

DECLARE_SHAPE_FN(rms_norm_bp) {
    auto inputShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape), shape::order(inputShape),
        shape::rank(inputShape), shape::shapeOf(inputShape)));
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
    auto inputShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape), shape::order(inputShape),
        shape::rank(inputShape), shape::shapeOf(inputShape)));
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
    auto inputShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape), shape::order(inputShape),
        shape::rank(inputShape), shape::shapeOf(inputShape)));
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
    auto sigmoid = input->transform(transform::Sigmoid);
    output->assign(*input * sigmoid);

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
    auto sigmoid = input->transform(transform::Sigmoid);
    auto sigmoidDeriv = sigmoid * (1.0f - sigmoid);
    auto grad = sigmoid + *input * sigmoidDeriv;

    gradIn->assign(*gradOut * grad);

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
// grouped_query_attention - Grouped Query Attention
#if NOT_EXCLUDED(OP_grouped_query_attention)
CUSTOM_OP_IMPL(grouped_query_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);   // [batch, seq_len, num_heads, head_dim]
    auto key = INPUT_VARIABLE(1);     // [batch, kv_len, num_kv_heads, head_dim]
    auto value = INPUT_VARIABLE(2);   // [batch, kv_len, num_kv_heads, head_dim]
    auto output = OUTPUT_VARIABLE(0);

    int numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : 8;
    int numKvHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : numHeads;
    bool isCausal = block.getIArguments()->size() > 2 ? INT_ARG(2) != 0 : true;

    auto headDim = query->sizeAt(-1);
    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) : 1.0f / std::sqrt(static_cast<float>(headDim));

    auto batch = query->sizeAt(0);
    auto seqLen = query->sizeAt(1);
    auto kvLen = key->sizeAt(1);

    // Reshape for attention computation
    auto qReshaped = query->reshape(query->ordering(), {batch * numHeads, seqLen, headDim});
    auto kReshaped = key->reshape(key->ordering(), {batch * numKvHeads, kvLen, headDim});
    auto vReshaped = value->reshape(value->ordering(), {batch * numKvHeads, kvLen, headDim});

    // Handle GQA by repeating KV heads
    int headsPerKv = numHeads / numKvHeads;

    // Compute attention for each batch
    auto outputBuf = output->bufferAsT<float>();
    auto queryBuf = query->bufferAsT<float>();
    auto keyBuf = key->bufferAsT<float>();
    auto valueBuf = value->bufferAsT<float>();

    // Simple nested loop implementation
    for (LongType b = 0; b < batch; ++b) {
        for (LongType h = 0; h < numHeads; ++h) {
            int kvHead = h / headsPerKv;

            for (LongType sq = 0; sq < seqLen; ++sq) {
                // Compute attention scores
                std::vector<float> scores(kvLen);
                float maxScore = -1e9f;

                for (LongType sk = 0; sk < kvLen; ++sk) {
                    if (isCausal && sk > sq) {
                        scores[sk] = -1e9f;
                        continue;
                    }

                    float score = 0.0f;
                    for (LongType d = 0; d < headDim; ++d) {
                        LongType qIdx = ((b * seqLen + sq) * numHeads + h) * headDim + d;
                        LongType kIdx = ((b * kvLen + sk) * numKvHeads + kvHead) * headDim + d;
                        score += queryBuf[qIdx] * keyBuf[kIdx];
                    }
                    scores[sk] = score * scale;
                    maxScore = std::max(maxScore, scores[sk]);
                }

                // Softmax
                float sumExp = 0.0f;
                for (LongType sk = 0; sk < kvLen; ++sk) {
                    scores[sk] = std::exp(scores[sk] - maxScore);
                    sumExp += scores[sk];
                }
                for (LongType sk = 0; sk < kvLen; ++sk) {
                    scores[sk] /= sumExp;
                }

                // Weighted sum of values
                for (LongType d = 0; d < headDim; ++d) {
                    float sum = 0.0f;
                    for (LongType sk = 0; sk < kvLen; ++sk) {
                        LongType vIdx = ((b * kvLen + sk) * numKvHeads + kvHead) * headDim + d;
                        sum += scores[sk] * valueBuf[vIdx];
                    }
                    LongType outIdx = ((b * seqLen + sq) * numHeads + h) * headDim + d;
                    outputBuf[outIdx] = sum;
                }
            }
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(grouped_query_attention) {
    auto queryShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(queryShape), shape::order(queryShape),
        shape::rank(queryShape), shape::shapeOf(queryShape)));
}

DECLARE_TYPES(grouped_query_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(grouped_query_attention_bp, 4, 3, false, 0, 0) {
    // Placeholder - actual implementation needed for training
    auto gradQ = OUTPUT_VARIABLE(0);
    auto gradK = OUTPUT_VARIABLE(1);
    auto gradV = OUTPUT_VARIABLE(2);

    gradQ->assign(0.0f);
    gradK->assign(0.0f);
    gradV->assign(0.0f);

    return Status::OK;
}

DECLARE_SHAPE_FN(grouped_query_attention_bp) {
    auto queryShape = inputShape->at(0);
    auto keyShape = inputShape->at(1);
    auto valueShape = inputShape->at(2);

    auto gradQShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(queryShape), shape::order(queryShape),
        shape::rank(queryShape), shape::shapeOf(queryShape));
    auto gradKShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(keyShape), shape::order(keyShape),
        shape::rank(keyShape), shape::shapeOf(keyShape));
    auto gradVShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(valueShape), shape::order(valueShape),
        shape::rank(valueShape), shape::shapeOf(valueShape));

    return SHAPELIST(gradQShape, gradKShape, gradVShape);
}

DECLARE_TYPES(grouped_query_attention_bp) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// flash_attention - Memory-efficient attention
#if NOT_EXCLUDED(OP_flash_attention)
CUSTOM_OP_IMPL(flash_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    bool isCausal = block.getIArguments()->size() > 0 ? INT_ARG(0) != 0 : true;
    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) : 1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));

    // For generic implementation, fall back to standard attention
    // Platform helpers will use optimized flash attention
    auto batch = query->sizeAt(0);
    auto seqLen = query->sizeAt(1);
    auto numHeads = query->sizeAt(2);
    auto headDim = query->sizeAt(3);
    auto kvLen = key->sizeAt(1);

    auto queryBuf = query->bufferAsT<float>();
    auto keyBuf = key->bufferAsT<float>();
    auto valueBuf = value->bufferAsT<float>();
    auto outputBuf = output->bufferAsT<float>();

    for (LongType b = 0; b < batch; ++b) {
        for (LongType h = 0; h < numHeads; ++h) {
            for (LongType sq = 0; sq < seqLen; ++sq) {
                std::vector<float> scores(kvLen);
                float maxScore = -1e9f;

                for (LongType sk = 0; sk < kvLen; ++sk) {
                    if (isCausal && sk > sq) {
                        scores[sk] = -1e9f;
                        continue;
                    }

                    float score = 0.0f;
                    for (LongType d = 0; d < headDim; ++d) {
                        LongType qIdx = ((b * seqLen + sq) * numHeads + h) * headDim + d;
                        LongType kIdx = ((b * kvLen + sk) * numHeads + h) * headDim + d;
                        score += queryBuf[qIdx] * keyBuf[kIdx];
                    }
                    scores[sk] = score * scale;
                    maxScore = std::max(maxScore, scores[sk]);
                }

                float sumExp = 0.0f;
                for (LongType sk = 0; sk < kvLen; ++sk) {
                    scores[sk] = std::exp(scores[sk] - maxScore);
                    sumExp += scores[sk];
                }
                for (LongType sk = 0; sk < kvLen; ++sk) {
                    scores[sk] /= sumExp;
                }

                for (LongType d = 0; d < headDim; ++d) {
                    float sum = 0.0f;
                    for (LongType sk = 0; sk < kvLen; ++sk) {
                        LongType vIdx = ((b * kvLen + sk) * numHeads + h) * headDim + d;
                        sum += scores[sk] * valueBuf[vIdx];
                    }
                    LongType outIdx = ((b * seqLen + sq) * numHeads + h) * headDim + d;
                    outputBuf[outIdx] = sum;
                }
            }
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(flash_attention) {
    auto queryShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(queryShape), shape::order(queryShape),
        shape::rank(queryShape), shape::shapeOf(queryShape)));
}

DECLARE_TYPES(flash_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(flash_attention_bp, 4, 3, false, 0, 0) {
    auto gradQ = OUTPUT_VARIABLE(0);
    auto gradK = OUTPUT_VARIABLE(1);
    auto gradV = OUTPUT_VARIABLE(2);

    gradQ->assign(0.0f);
    gradK->assign(0.0f);
    gradV->assign(0.0f);

    return Status::OK;
}

DECLARE_SHAPE_FN(flash_attention_bp) {
    auto queryShape = inputShape->at(0);
    auto keyShape = inputShape->at(1);
    auto valueShape = inputShape->at(2);

    return SHAPELIST(
        ConstantShapeHelper::getInstance().createShapeInfo(
            ArrayOptions::dataType(queryShape), shape::order(queryShape),
            shape::rank(queryShape), shape::shapeOf(queryShape)),
        ConstantShapeHelper::getInstance().createShapeInfo(
            ArrayOptions::dataType(keyShape), shape::order(keyShape),
            shape::rank(keyShape), shape::shapeOf(keyShape)),
        ConstantShapeHelper::getInstance().createShapeInfo(
            ArrayOptions::dataType(valueShape), shape::order(valueShape),
            shape::rank(valueShape), shape::shapeOf(valueShape)));
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

    // Update with new keys/values at position
    auto newSeqLen = newKeys->sizeAt(1);

    // Copy new values into cache at the specified position
    for (LongType i = 0; i < newSeqLen; ++i) {
        auto keySlice = newKeys->subarray({{}, {i, i + 1}});
        auto valueSlice = newValues->subarray({{}, {i, i + 1}});

        auto outKeySlice = outputKeyCache->subarray({{}, {startPos + i, startPos + i + 1}});
        auto outValueSlice = outputValueCache->subarray({{}, {startPos + i, startPos + i + 1}});

        outKeySlice.assign(keySlice);
        outValueSlice.assign(valueSlice);
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(kv_cache_update) {
    auto keyCacheShape = inputShape->at(0);
    auto valueCacheShape = inputShape->at(1);

    return SHAPELIST(
        ConstantShapeHelper::getInstance().createShapeInfo(
            ArrayOptions::dataType(keyCacheShape), shape::order(keyCacheShape),
            shape::rank(keyCacheShape), shape::shapeOf(keyCacheShape)),
        ConstantShapeHelper::getInstance().createShapeInfo(
            ArrayOptions::dataType(valueCacheShape), shape::order(valueCacheShape),
            shape::rank(valueCacheShape), shape::shapeOf(valueCacheShape)));
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
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(scoresShape), shape::order(scoresShape),
        shape::rank(scoresShape), shape::shapeOf(scoresShape)));
}

DECLARE_TYPES(apply_alibi) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

//////////////////////////////////////////////////////////////////////////
// sliding_window_attention - Sliding Window Attention
#if NOT_EXCLUDED(OP_sliding_window_attention)
CUSTOM_OP_IMPL(sliding_window_attention, 3, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    int windowSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 4096;
    int numHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : static_cast<int>(query->sizeAt(2));
    int numKvHeads = block.getIArguments()->size() > 2 ? INT_ARG(2) : numHeads;
    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) : 1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));

    auto batch = query->sizeAt(0);
    auto seqLen = query->sizeAt(1);
    auto headDim = query->sizeAt(3);
    auto kvLen = key->sizeAt(1);

    int headsPerKv = numHeads / numKvHeads;

    auto queryBuf = query->bufferAsT<float>();
    auto keyBuf = key->bufferAsT<float>();
    auto valueBuf = value->bufferAsT<float>();
    auto outputBuf = output->bufferAsT<float>();

    for (LongType b = 0; b < batch; ++b) {
        for (LongType h = 0; h < numHeads; ++h) {
            int kvHead = h / headsPerKv;

            for (LongType sq = 0; sq < seqLen; ++sq) {
                // Sliding window: only attend to positions within window
                LongType windowStart = std::max(0LL, sq - windowSize + 1);
                LongType windowEnd = sq + 1;

                std::vector<float> scores(kvLen, -1e9f);
                float maxScore = -1e9f;

                for (LongType sk = windowStart; sk < windowEnd && sk < kvLen; ++sk) {
                    float score = 0.0f;
                    for (LongType d = 0; d < headDim; ++d) {
                        LongType qIdx = ((b * seqLen + sq) * numHeads + h) * headDim + d;
                        LongType kIdx = ((b * kvLen + sk) * numKvHeads + kvHead) * headDim + d;
                        score += queryBuf[qIdx] * keyBuf[kIdx];
                    }
                    scores[sk] = score * scale;
                    maxScore = std::max(maxScore, scores[sk]);
                }

                float sumExp = 0.0f;
                for (LongType sk = 0; sk < kvLen; ++sk) {
                    scores[sk] = std::exp(scores[sk] - maxScore);
                    sumExp += scores[sk];
                }
                for (LongType sk = 0; sk < kvLen; ++sk) {
                    scores[sk] /= sumExp;
                }

                for (LongType d = 0; d < headDim; ++d) {
                    float sum = 0.0f;
                    for (LongType sk = windowStart; sk < windowEnd && sk < kvLen; ++sk) {
                        LongType vIdx = ((b * kvLen + sk) * numKvHeads + kvHead) * headDim + d;
                        sum += scores[sk] * valueBuf[vIdx];
                    }
                    LongType outIdx = ((b * seqLen + sq) * numHeads + h) * headDim + d;
                    outputBuf[outIdx] = sum;
                }
            }
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(sliding_window_attention) {
    auto queryShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(queryShape), shape::order(queryShape),
        shape::rank(queryShape), shape::shapeOf(queryShape)));
}

DECLARE_TYPES(sliding_window_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

}  // namespace ops
}  // namespace sd

#endif
