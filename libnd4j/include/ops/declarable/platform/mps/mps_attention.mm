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
// Metal Performance Shaders - Attention mechanism operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"
#include <cmath>
#include <algorithm>

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Scaled Dot-Product Attention: softmax(Q * K^T / sqrt(d_k)) * V
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(dot_product_attention, ENGINE_CPU) {
    auto queries = INPUT_VARIABLE(0);  // [batch, heads, seqQ, depth]
    auto keys = INPUT_VARIABLE(1);     // [batch, heads, seqK, depth]
    auto values = INPUT_VARIABLE(2);   // [batch, heads, seqK, depthV]
    // Optional mask at INPUT_VARIABLE(3)

    auto output = OUTPUT_VARIABLE(0);   // [batch, heads, seqQ, depthV]
    // Optional attention weights at OUTPUT_VARIABLE(1)

    if (queries->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* qPtr = queries->bufferAsT<float>();
        const float* kPtr = keys->bufferAsT<float>();
        const float* vPtr = values->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        LongType batch = queries->sizeAt(0);
        LongType heads = queries->sizeAt(1);
        LongType seqQ = queries->sizeAt(2);
        LongType depth = queries->sizeAt(3);
        LongType seqK = keys->sizeAt(2);
        LongType depthV = values->sizeAt(3);

        float scale = 1.0f / sqrtf((float)depth);

        // Check for optional mask
        const float* maskPtr = nullptr;
        bool hasMask = block.width() > 3;
        if (hasMask) {
            auto mask = INPUT_VARIABLE(3);
            if (!mask->isEmpty()) {
                maskPtr = mask->bufferAsT<float>();
            }
        }

        // Process each batch and head
        for (LongType b = 0; b < batch; b++) {
            for (LongType h = 0; h < heads; h++) {
                // Compute attention scores: Q * K^T
                std::vector<float> scores(seqQ * seqK);

                for (LongType q = 0; q < seqQ; q++) {
                    for (LongType k = 0; k < seqK; k++) {
                        float dot = 0.0f;
                        for (LongType d = 0; d < depth; d++) {
                            LongType qIdx = b * heads * seqQ * depth + h * seqQ * depth + q * depth + d;
                            LongType kIdx = b * heads * seqK * depth + h * seqK * depth + k * depth + d;
                            dot += qPtr[qIdx] * kPtr[kIdx];
                        }
                        scores[q * seqK + k] = dot * scale;
                    }
                }

                // Apply mask if present (add large negative value to masked positions)
                if (maskPtr != nullptr) {
                    for (LongType q = 0; q < seqQ; q++) {
                        for (LongType k = 0; k < seqK; k++) {
                            LongType maskIdx = b * seqQ * seqK + q * seqK + k;  // Simplified mask indexing
                            if (maskPtr[maskIdx] == 0.0f) {
                                scores[q * seqK + k] = -1e9f;
                            }
                        }
                    }
                }

                // Softmax over K dimension
                for (LongType q = 0; q < seqQ; q++) {
                    float maxVal = scores[q * seqK];
                    for (LongType k = 1; k < seqK; k++) {
                        if (scores[q * seqK + k] > maxVal) {
                            maxVal = scores[q * seqK + k];
                        }
                    }

                    float sum = 0.0f;
                    for (LongType k = 0; k < seqK; k++) {
                        scores[q * seqK + k] = expf(scores[q * seqK + k] - maxVal);
                        sum += scores[q * seqK + k];
                    }

                    for (LongType k = 0; k < seqK; k++) {
                        scores[q * seqK + k] /= sum;
                    }
                }

                // Compute output: attention_weights * V
                for (LongType q = 0; q < seqQ; q++) {
                    for (LongType dv = 0; dv < depthV; dv++) {
                        float sum = 0.0f;
                        for (LongType k = 0; k < seqK; k++) {
                            LongType vIdx = b * heads * seqK * depthV + h * seqK * depthV + k * depthV + dv;
                            sum += scores[q * seqK + k] * vPtr[vIdx];
                        }
                        LongType outIdx = b * heads * seqQ * depthV + h * seqQ * depthV + q * depthV + dv;
                        outPtr[outIdx] = sum;
                    }
                }

                // Optionally output attention weights
                if (block.outputWidth() > 1) {
                    auto attentionWeights = OUTPUT_VARIABLE(1);
                    float* awPtr = attentionWeights->bufferAsT<float>();
                    for (LongType q = 0; q < seqQ; q++) {
                        for (LongType k = 0; k < seqK; k++) {
                            LongType awIdx = b * heads * seqQ * seqK + h * seqQ * seqK + q * seqK + k;
                            awPtr[awIdx] = scores[q * seqK + k];
                        }
                    }
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(dot_product_attention, ENGINE_CPU) {
    auto queries = INPUT_VARIABLE(0);

    Requirements req("MPS DOT_PRODUCT_ATTENTION OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(queries->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*queries), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Multi-Head Attention
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(multi_head_dot_product_attention, ENGINE_CPU) {
    auto queries = INPUT_VARIABLE(0);  // [batch, seqQ, embedDim]
    auto keys = INPUT_VARIABLE(1);     // [batch, seqK, embedDim]
    auto values = INPUT_VARIABLE(2);   // [batch, seqK, embedDim]
    auto Wq = INPUT_VARIABLE(3);       // [embedDim, numHeads * headDim]
    auto Wk = INPUT_VARIABLE(4);       // [embedDim, numHeads * headDim]
    auto Wv = INPUT_VARIABLE(5);       // [embedDim, numHeads * headDim]
    auto Wo = INPUT_VARIABLE(6);       // [numHeads * headDim, embedDim]
    // Optional mask at INPUT_VARIABLE(7)

    auto output = OUTPUT_VARIABLE(0);  // [batch, seqQ, embedDim]

    if (queries->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int numHeads = INT_ARG(0);
        bool normalization = block.getIArguments()->size() > 1 ? (INT_ARG(1) != 0) : true;

        const float* qPtr = queries->bufferAsT<float>();
        const float* kPtr = keys->bufferAsT<float>();
        const float* vPtr = values->bufferAsT<float>();
        const float* WqPtr = Wq->bufferAsT<float>();
        const float* WkPtr = Wk->bufferAsT<float>();
        const float* WvPtr = Wv->bufferAsT<float>();
        const float* WoPtr = Wo->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        LongType batch = queries->sizeAt(0);
        LongType seqQ = queries->sizeAt(1);
        LongType embedDim = queries->sizeAt(2);
        LongType seqK = keys->sizeAt(1);
        LongType projDim = Wq->sizeAt(1);
        LongType headDim = projDim / numHeads;

        float scale = 1.0f / sqrtf((float)headDim);

        // Project Q, K, V
        std::vector<float> projQ(batch * seqQ * projDim);
        std::vector<float> projK(batch * seqK * projDim);
        std::vector<float> projV(batch * seqK * projDim);

        // Q projection: Q * Wq
        for (LongType b = 0; b < batch; b++) {
            for (LongType s = 0; s < seqQ; s++) {
                for (LongType p = 0; p < projDim; p++) {
                    float sum = 0.0f;
                    for (LongType e = 0; e < embedDim; e++) {
                        sum += qPtr[b * seqQ * embedDim + s * embedDim + e] * WqPtr[e * projDim + p];
                    }
                    projQ[b * seqQ * projDim + s * projDim + p] = sum;
                }
            }
        }

        // K projection: K * Wk
        for (LongType b = 0; b < batch; b++) {
            for (LongType s = 0; s < seqK; s++) {
                for (LongType p = 0; p < projDim; p++) {
                    float sum = 0.0f;
                    for (LongType e = 0; e < embedDim; e++) {
                        sum += kPtr[b * seqK * embedDim + s * embedDim + e] * WkPtr[e * projDim + p];
                    }
                    projK[b * seqK * projDim + s * projDim + p] = sum;
                }
            }
        }

        // V projection: V * Wv
        for (LongType b = 0; b < batch; b++) {
            for (LongType s = 0; s < seqK; s++) {
                for (LongType p = 0; p < projDim; p++) {
                    float sum = 0.0f;
                    for (LongType e = 0; e < embedDim; e++) {
                        sum += vPtr[b * seqK * embedDim + s * embedDim + e] * WvPtr[e * projDim + p];
                    }
                    projV[b * seqK * projDim + s * projDim + p] = sum;
                }
            }
        }

        // Attention per head, then concatenate
        std::vector<float> attnOut(batch * seqQ * projDim);

        for (LongType b = 0; b < batch; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Compute attention scores for this head
                std::vector<float> scores(seqQ * seqK);

                for (LongType q = 0; q < seqQ; q++) {
                    for (LongType k = 0; k < seqK; k++) {
                        float dot = 0.0f;
                        for (LongType d = 0; d < headDim; d++) {
                            LongType qIdx = b * seqQ * projDim + q * projDim + h * headDim + d;
                            LongType kIdx = b * seqK * projDim + k * projDim + h * headDim + d;
                            dot += projQ[qIdx] * projK[kIdx];
                        }
                        scores[q * seqK + k] = dot * scale;
                    }
                }

                // Softmax
                for (LongType q = 0; q < seqQ; q++) {
                    float maxVal = scores[q * seqK];
                    for (LongType k = 1; k < seqK; k++) {
                        if (scores[q * seqK + k] > maxVal) {
                            maxVal = scores[q * seqK + k];
                        }
                    }

                    float sum = 0.0f;
                    for (LongType k = 0; k < seqK; k++) {
                        scores[q * seqK + k] = expf(scores[q * seqK + k] - maxVal);
                        sum += scores[q * seqK + k];
                    }

                    for (LongType k = 0; k < seqK; k++) {
                        scores[q * seqK + k] /= sum;
                    }
                }

                // Apply attention to values
                for (LongType q = 0; q < seqQ; q++) {
                    for (LongType d = 0; d < headDim; d++) {
                        float sum = 0.0f;
                        for (LongType k = 0; k < seqK; k++) {
                            LongType vIdx = b * seqK * projDim + k * projDim + h * headDim + d;
                            sum += scores[q * seqK + k] * projV[vIdx];
                        }
                        LongType outIdx = b * seqQ * projDim + q * projDim + h * headDim + d;
                        attnOut[outIdx] = sum;
                    }
                }
            }
        }

        // Output projection: concat(heads) * Wo
        for (LongType b = 0; b < batch; b++) {
            for (LongType s = 0; s < seqQ; s++) {
                for (LongType e = 0; e < embedDim; e++) {
                    float sum = 0.0f;
                    for (LongType p = 0; p < projDim; p++) {
                        sum += attnOut[b * seqQ * projDim + s * projDim + p] * WoPtr[p * embedDim + e];
                    }
                    outPtr[b * seqQ * embedDim + s * embedDim + e] = sum;
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(multi_head_dot_product_attention, ENGINE_CPU) {
    auto queries = INPUT_VARIABLE(0);

    Requirements req("MPS MULTI_HEAD_DOT_PRODUCT_ATTENTION OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(queries->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*queries), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Additive Attention (Bahdanau attention)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(additive_attention, ENGINE_CPU) {
    auto queries = INPUT_VARIABLE(0);  // [batch, seqQ, depth]
    auto keys = INPUT_VARIABLE(1);     // [batch, seqK, depth]
    auto values = INPUT_VARIABLE(2);   // [batch, seqK, depthV]
    auto Wq = INPUT_VARIABLE(3);       // [depth, attnDim]
    auto Wk = INPUT_VARIABLE(4);       // [depth, attnDim]
    auto v = INPUT_VARIABLE(5);        // [attnDim]

    auto output = OUTPUT_VARIABLE(0);  // [batch, seqQ, depthV]

    if (queries->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* qPtr = queries->bufferAsT<float>();
        const float* kPtr = keys->bufferAsT<float>();
        const float* valsPtr = values->bufferAsT<float>();
        const float* WqPtr = Wq->bufferAsT<float>();
        const float* WkPtr = Wk->bufferAsT<float>();
        const float* vPtr = v->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        LongType batch = queries->sizeAt(0);
        LongType seqQ = queries->sizeAt(1);
        LongType depth = queries->sizeAt(2);
        LongType seqK = keys->sizeAt(1);
        LongType depthV = values->sizeAt(2);
        LongType attnDim = Wq->sizeAt(1);

        for (LongType b = 0; b < batch; b++) {
            for (LongType q = 0; q < seqQ; q++) {
                // Compute attention scores
                std::vector<float> scores(seqK);

                for (LongType k = 0; k < seqK; k++) {
                    // score = v^T * tanh(Wq * q + Wk * k)
                    float score = 0.0f;
                    for (LongType a = 0; a < attnDim; a++) {
                        float qProj = 0.0f, kProj = 0.0f;
                        for (LongType d = 0; d < depth; d++) {
                            qProj += qPtr[b * seqQ * depth + q * depth + d] * WqPtr[d * attnDim + a];
                            kProj += kPtr[b * seqK * depth + k * depth + d] * WkPtr[d * attnDim + a];
                        }
                        score += vPtr[a] * tanhf(qProj + kProj);
                    }
                    scores[k] = score;
                }

                // Softmax
                float maxVal = scores[0];
                for (LongType k = 1; k < seqK; k++) {
                    if (scores[k] > maxVal) maxVal = scores[k];
                }

                float sum = 0.0f;
                for (LongType k = 0; k < seqK; k++) {
                    scores[k] = expf(scores[k] - maxVal);
                    sum += scores[k];
                }

                for (LongType k = 0; k < seqK; k++) {
                    scores[k] /= sum;
                }

                // Weighted sum of values
                for (LongType dv = 0; dv < depthV; dv++) {
                    float weighted = 0.0f;
                    for (LongType k = 0; k < seqK; k++) {
                        weighted += scores[k] * valsPtr[b * seqK * depthV + k * depthV + dv];
                    }
                    outPtr[b * seqQ * depthV + q * depthV + dv] = weighted;
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(additive_attention, ENGINE_CPU) {
    auto queries = INPUT_VARIABLE(0);

    Requirements req("MPS ADDITIVE_ATTENTION OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(queries->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*queries), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Self-Attention (single sequence, no separate K/V)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(self_attention, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);      // [batch, seq, embed]
    auto Wqkv = INPUT_VARIABLE(1);   // [embed, 3*embed] - combined Q,K,V projection
    auto Wo = INPUT_VARIABLE(2);     // [embed, embed]

    auto output = OUTPUT_VARIABLE(0);  // [batch, seq, embed]

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int numHeads = INT_ARG(0);

        const float* xPtr = x->bufferAsT<float>();
        const float* WqkvPtr = Wqkv->bufferAsT<float>();
        const float* WoPtr = Wo->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        LongType batch = x->sizeAt(0);
        LongType seq = x->sizeAt(1);
        LongType embed = x->sizeAt(2);
        LongType headDim = embed / numHeads;

        float scale = 1.0f / sqrtf((float)headDim);

        // Project to Q, K, V
        std::vector<float> qkv(batch * seq * 3 * embed);

        for (LongType b = 0; b < batch; b++) {
            for (LongType s = 0; s < seq; s++) {
                for (LongType p = 0; p < 3 * embed; p++) {
                    float sum = 0.0f;
                    for (LongType e = 0; e < embed; e++) {
                        sum += xPtr[b * seq * embed + s * embed + e] * WqkvPtr[e * 3 * embed + p];
                    }
                    qkv[b * seq * 3 * embed + s * 3 * embed + p] = sum;
                }
            }
        }

        // Multi-head self-attention
        std::vector<float> attnOut(batch * seq * embed);

        for (LongType b = 0; b < batch; b++) {
            for (int h = 0; h < numHeads; h++) {
                std::vector<float> scores(seq * seq);

                // Compute Q * K^T / sqrt(d_k)
                for (LongType i = 0; i < seq; i++) {
                    for (LongType j = 0; j < seq; j++) {
                        float dot = 0.0f;
                        for (LongType d = 0; d < headDim; d++) {
                            LongType qIdx = b * seq * 3 * embed + i * 3 * embed + h * headDim + d;
                            LongType kIdx = b * seq * 3 * embed + j * 3 * embed + embed + h * headDim + d;
                            dot += qkv[qIdx] * qkv[kIdx];
                        }
                        scores[i * seq + j] = dot * scale;
                    }
                }

                // Causal mask (optional, for decoder self-attention)
                bool causal = block.getIArguments()->size() > 1 ? (INT_ARG(1) != 0) : false;
                if (causal) {
                    for (LongType i = 0; i < seq; i++) {
                        for (LongType j = i + 1; j < seq; j++) {
                            scores[i * seq + j] = -1e9f;
                        }
                    }
                }

                // Softmax
                for (LongType i = 0; i < seq; i++) {
                    float maxVal = scores[i * seq];
                    for (LongType j = 1; j < seq; j++) {
                        if (scores[i * seq + j] > maxVal) maxVal = scores[i * seq + j];
                    }

                    float sum = 0.0f;
                    for (LongType j = 0; j < seq; j++) {
                        scores[i * seq + j] = expf(scores[i * seq + j] - maxVal);
                        sum += scores[i * seq + j];
                    }

                    for (LongType j = 0; j < seq; j++) {
                        scores[i * seq + j] /= sum;
                    }
                }

                // Apply to V
                for (LongType i = 0; i < seq; i++) {
                    for (LongType d = 0; d < headDim; d++) {
                        float sum = 0.0f;
                        for (LongType j = 0; j < seq; j++) {
                            LongType vIdx = b * seq * 3 * embed + j * 3 * embed + 2 * embed + h * headDim + d;
                            sum += scores[i * seq + j] * qkv[vIdx];
                        }
                        attnOut[b * seq * embed + i * embed + h * headDim + d] = sum;
                    }
                }
            }
        }

        // Output projection
        for (LongType b = 0; b < batch; b++) {
            for (LongType s = 0; s < seq; s++) {
                for (LongType e = 0; e < embed; e++) {
                    float sum = 0.0f;
                    for (LongType p = 0; p < embed; p++) {
                        sum += attnOut[b * seq * embed + s * embed + p] * WoPtr[p * embed + e];
                    }
                    outPtr[b * seq * embed + s * embed + e] = sum;
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(self_attention, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SELF_ATTENTION OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
