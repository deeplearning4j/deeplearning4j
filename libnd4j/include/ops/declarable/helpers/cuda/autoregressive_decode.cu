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

#include <system/op_boilerplate.h>
#include <helpers/logger.h>
#include <ops/declarable/helpers/autoregressive_decode.h>
#include <ops/declarable/helpers/token_sample.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <graph/Context.h>
#include <graph/NativeDynamicShapePlan.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// ─── CUDA Kernels ────────────────────────────────────────────────────────────

/**
 * CUDA kernel: look up a single row from the embedding table.
 *
 * Given embeddingTable [vocabSize, hidden] and a token ID, copies
 * embeddingTable[tokenId, :] into outputEmbed [1, 1, hidden].
 *
 * One block, blockDim.x threads — each thread copies hidden/blockDim.x elements.
 */
template <typename T>
static SD_KERNEL void embedLookupKernel(const void* vEmbTable,
                                         void* vOutput,
                                         LongType tokenId,
                                         LongType hidden,
                                         LongType tableRowStride) {
    auto embTable = reinterpret_cast<const T*>(vEmbTable);
    auto output = reinterpret_cast<T*>(vOutput);

    LongType baseOffset = tokenId * tableRowStride;
    for (LongType i = threadIdx.x; i < hidden; i += blockDim.x) {
        output[i] = embTable[baseOffset + i];
    }
}

/**
 * Launcher for embedLookupKernel — called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void embedLookupLauncher(const cudaStream_t* stream, const void* embTable,
                                 void* output, LongType tokenId,
                                 LongType hidden, LongType tableRowStride) {
    embedLookupKernel<T><<<1, 256, 0, *stream>>>(embTable, output, tokenId, hidden, tableRowStride);
}

/**
 * CUDA kernel: update attention mask for the next decode step.
 *
 * Sets mask[position] = 1.0 (unmask the new position).
 * The mask is [1, 1, 1, maxKvLen] for single-token decode.
 */
template <typename T>
static SD_KERNEL void updateAttentionMaskKernel(void* vMask,
                                                  LongType position,
                                                  LongType maxKvLen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto mask = reinterpret_cast<T*>(vMask);
        if (position < maxKvLen) {
            mask[position] = static_cast<T>(1);
        }
    }
}

/**
 * Launcher for updateAttentionMaskKernel — called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void updateAttentionMaskLauncher(const cudaStream_t* stream,
                                         void* vMask,
                                         LongType position,
                                         LongType maxKvLen) {
    updateAttentionMaskKernel<T><<<1, 1, 0, *stream>>>(vMask, position, maxKvLen);
}

/**
 * CUDA kernel: update position_ids for the next decode step.
 *
 * Sets positionIds[0] = newPosition.
 */
static SD_KERNEL void updatePositionIdsKernel(void* vPositionIds,
                                                LongType newPosition) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto posIds = reinterpret_cast<LongType*>(vPositionIds);
        posIds[0] = newPosition;
    }
}

/**
 * CUDA kernel: update input_ids for the next decode step.
 *
 * Sets inputIds[0] = newTokenId.
 */
static SD_KERNEL void updateInputIdsKernel(void* vInputIds,
                                             LongType newTokenId) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto inputIds = reinterpret_cast<LongType*>(vInputIds);
        inputIds[0] = newTokenId;
    }
}

/**
 * CUDA kernel: update causal mask for the next decode step.
 *
 * Sets causalMask[position] = 0.0f (unmask the new position).
 * The causal mask is [1, 1, 1, maskLen] FLOAT for single-token decode,
 * filled with MASK_FILL (-3.4028235e+38f) for masked positions and 0.0f
 * for unmasked positions. Each decode step unmasks one more position.
 */
template <typename T>
static SD_KERNEL void updateCausalMaskKernel(void* vMask,
                                               LongType position,
                                               LongType maskLen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto mask = reinterpret_cast<T*>(vMask);
        if (position < maskLen) {
            mask[position] = static_cast<T>(0);
        }
    }
}

/**
 * Launcher for updateCausalMaskKernel — called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void updateCausalMaskLauncher(const cudaStream_t* stream,
                                      void* vMask,
                                      LongType position,
                                      LongType maskLen) {
    updateCausalMaskKernel<T><<<1, 1, 0, *stream>>>(vMask, position, maskLen);
}

/**
 * CUDA kernel: build initial attention mask from prefill length.
 *
 * Sets mask[0..prefillSeqLen-1] = 1, rest stays 0.
 * Mask is pre-zeroed by the caller.
 */
template <typename T>
static SD_KERNEL void buildInitialMaskKernel(void* vMask, LongType prefillSeqLen, LongType maxKvLen) {
    LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < prefillSeqLen && idx < maxKvLen) {
        reinterpret_cast<T*>(vMask)[idx] = static_cast<T>(1);
    }
}

/**
 * Launcher for buildInitialMaskKernel.
 */
template <typename T>
static void buildInitialMaskLauncher(const cudaStream_t* stream, void* vMask,
                                      LongType prefillSeqLen, LongType maxKvLen) {
    int threads = 256;
    int blocks = (prefillSeqLen + threads - 1) / threads;
    buildInitialMaskKernel<T><<<blocks, threads, 0, *stream>>>(vMask, prefillSeqLen, maxKvLen);
}

// ─── Argmax helper (greedy decode) ───────────────────────────────────────────

/**
 * CUDA kernel: find argmax over a float/half row [vocabSize].
 * Writes the index to output[0] as INT64.
 *
 * Block-level reduction using shared memory.
 */
template <typename T>
static SD_KERNEL void argmaxKernel(const void* vLogits, void* vOutput, LongType vocabSize) {
    extern __shared__ char smem[];
    auto sMaxVal = reinterpret_cast<T*>(smem);
    auto sMaxIdx = reinterpret_cast<LongType*>(smem + blockDim.x * sizeof(T));

    auto logits = reinterpret_cast<const T*>(vLogits);
    auto output = reinterpret_cast<LongType*>(vOutput);

    T localMax = static_cast<T>(-1e30);
    LongType localIdx = 0;

    for (LongType i = threadIdx.x; i < vocabSize; i += blockDim.x) {
        T val = logits[i];
        if (val > localMax) {
            localMax = val;
            localIdx = i;
        }
    }

    sMaxVal[threadIdx.x] = localMax;
    sMaxIdx[threadIdx.x] = localIdx;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sMaxVal[threadIdx.x + stride] > sMaxVal[threadIdx.x]) {
                sMaxVal[threadIdx.x] = sMaxVal[threadIdx.x + stride];
                sMaxIdx[threadIdx.x] = sMaxIdx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[0] = sMaxIdx[0];
    }
}

/**
 * Launcher for argmaxKernel — called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void argmaxLauncher(const cudaStream_t* stream, const void* logitsPtr,
                            void* outputPtr, LongType vocabSize) {
    int threads = 256;
    int smemSize = threads * (sizeof(T) + sizeof(LongType));
    argmaxKernel<T><<<1, threads, smemSize, *stream>>>(logitsPtr, outputPtr, vocabSize);
}

// ─── Main Implementation ─────────────────────────────────────────────────────

void autoregressiveDecodeCuda(
    NDArray* prefillEmbeddings,
    NDArray* embeddingTable,
    NDArray* inputIds,
    NDArray* attentionMask,
    NDArray* positionIds,
    NDArray** staticKvBuffers,
    int numKvPairs,
    NDArray* generatedTokenIds,
    NDArray* tokenCount,
    NDArray* timingInfo,
    int maxNewTokens,
    int prefillSeqLen,
    const std::vector<int>& stopTokenIds,
    double temperature,
    int topK,
    double topP,
    LaunchContext* context,
    AutoregressiveDecodeConfig* config) {

    auto stream = context->getCudaStream();

    // Initialize outputs
    LongType zero = 0;
    float zeroF = 0.0f;
    generatedTokenIds->assign(zero);
    tokenCount->assign(zero);
    timingInfo->assign(zeroF);

    // Validate that we have a plan to execute — hard error, not silent return.
    REQUIRE_TRUE(config != nullptr && config->planHandle != nullptr, 0,
                 "autoregressive_decode: no plan handle provided. "
                 "The Java side MUST pass a compiled NativeDynamicShapePlan via config->planHandle. "
                 "config=%p planHandle=%p",
                 config, config ? config->planHandle : nullptr);

    auto plan = config->planHandle;

    // ── Timing ──
    std::vector<double> stepTimesMs;
    stepTimesMs.reserve(maxNewTokens);
    auto loopStart = std::chrono::high_resolution_clock::now();

    // ── Internal state ──
    LongType currentPosition = static_cast<LongType>(prefillSeqLen);
    auto hidden = embeddingTable->sizeAt(1);
    auto vocabSize = embeddingTable->sizeAt(0);
    auto embTableRowStride = embeddingTable->strideAt(0);

    // ── Build internal attention mask if not provided ──
    // Shape: [1, 1, 1, maxKvLen] — single-token decode mask
    NDArray* internalMask = nullptr;
    LongType maxKvLen = 0;
    if (attentionMask != nullptr) {
        maxKvLen = attentionMask->sizeAt(-1);
    } else {
        // Allocate: maxKvLen = prefillSeqLen + maxNewTokens
        maxKvLen = prefillSeqLen + maxNewTokens;
        std::vector<LongType> maskShape = {1, 1, 1, maxKvLen};
        internalMask = NDArrayFactory::create('c', maskShape, DataType::FLOAT32, context);
        internalMask->assign(zeroF);
        // Fill prefill positions
        NDArray::prepareSpecialUse({internalMask}, {});
        BUILD_SINGLE_SELECTOR(internalMask->dataType(), buildInitialMaskLauncher,
                              (stream, internalMask->specialBuffer(), prefillSeqLen, maxKvLen),
                              SD_COMMON_TYPES);
        NDArray::registerSpecialUse({internalMask}, {});
        attentionMask = internalMask;
    }

    // ── Build internal position_ids if not provided ──
    // Shape: [1, 1] — single-token decode
    NDArray* internalPosIds = nullptr;
    if (positionIds == nullptr) {
        std::vector<LongType> posShape = {1, 1};
        internalPosIds = NDArrayFactory::create('c', posShape, DataType::INT64, context);
        internalPosIds->p(0, static_cast<LongType>(prefillSeqLen));
        positionIds = internalPosIds;
    }

    // ── Working buffers ──
    // Reuse prefillEmbeddings (Java's decodeEmbeddings [1,1,hidden]) for embed lookup.
    // CRITICAL: Do NOT allocate a new NDArray — the CUDA graph was captured with
    // prefillEmbeddings' device address as the embeddings ext input. Using a new
    // allocation would change the address, causing externalAddrsMatch() to fail,
    // which forces fallback to phaseReplay (broken ext input sync → degenerate output).
    NDArray* decodeEmbedding = prefillEmbeddings;

    // Token sample output: single INT64 scalar
    std::vector<LongType> sampleShape = {1};
    NDArray* sampledToken = NDArrayFactory::create('c', sampleShape, DataType::INT64, context);

    // Logits slice: [vocabSize] — last-position logits from plan output
    // We'll point into the plan's output buffer directly when possible

    int tokensGenerated = 0;

    // ── Get plan's external inputs from the persistent OpaqueContext ──
    // The Java DynamicShapePlanExecutor caches an OpaqueContext with all ext inputs
    // registered via setGraphContextInputArray(). That context persists across calls.
    // We read NDArray* pointers from it via ctx->array(i).
    auto* extCtx = reinterpret_cast<graph::Context*>(config->extInputContext);
    int numExtInputs = config->numPlanExternalInputs;

    // Build ext inputs array from the context's registered inputs
    std::vector<NDArray*> extInputsVec(numExtInputs);
    if (extCtx != nullptr) {
        for (int i = 0; i < numExtInputs; i++) {
            extInputsVec[i] = extCtx->array(i);
        }
    } else if (config->planExternalInputs != nullptr) {
        // Fallback: use directly passed array (legacy path)
        for (int i = 0; i < numExtInputs; i++) {
            extInputsVec[i] = config->planExternalInputs[i];
        }
    }
    NDArray** extInputs = extInputsVec.data();

    // ── Extract causal mask from ext inputs (if present) ──
    // The causal mask is a plan external input at config->causalMaskExtIdx.
    // It's [1, 1, 1, maskLen] FLOAT, filled with MASK_FILL for masked positions.
    // We need to update it per step: unmask position currentPosition with 0.0f.
    NDArray* causalMask = nullptr;
    LongType causalMaskLen = 0;
    if (config->causalMaskExtIdx >= 0 && config->causalMaskExtIdx < numExtInputs) {
        causalMask = extInputsVec[config->causalMaskExtIdx];
        if (causalMask != nullptr) {
            causalMaskLen = causalMask->sizeAt(-1);
        }
    }

    // ── Extract attn_mask_reformat from ext inputs (if present) ──
    // The attn_mask_reformat override bypasses the model's internal subgraph
    // which produces incorrect masks for padded static-KV decode. We delta-update
    // it each step just like the causal mask.
    NDArray* attnMaskReformat = nullptr;
    LongType attnMaskReformatLen = 0;
    if (config->attnMaskReformatExtIdx >= 0 && config->attnMaskReformatExtIdx < numExtInputs) {
        attnMaskReformat = extInputsVec[config->attnMaskReformatExtIdx];
        if (attnMaskReformat != nullptr) {
            attnMaskReformatLen = attnMaskReformat->sizeAt(-1);
        }
    }

    // Plan outputs: allocate array for plan to fill
    int numPlanOutputs = plan->getNumRequestedOutputs();
    std::vector<NDArray*> planOutputsVec(numPlanOutputs, nullptr);
    NDArray** planOutputs = planOutputsVec.data();

    REQUIRE_TRUE(extCtx != nullptr || config->planExternalInputs != nullptr, 0,
                 "autoregressive_decode: no external input source. "
                 "Either extInputContext (OpaqueContext*) or planExternalInputs (NDArray**) "
                 "must be non-null. Both are null — cannot wire plan inputs.");

    for (int step = 0; step < maxNewTokens; step++) {
        auto stepStart = std::chrono::high_resolution_clock::now();

        // ── Step 1: Update plan external inputs for this decode step ──
        // decodeEmbedding IS prefillEmbeddings (same NDArray, same device address).
        // The embed lookup kernel writes into it in-place each step, keeping the
        // device address stable for CUDA graph replay (externalAddrsMatch).
        if (config->embeddingsExtIdx >= 0 && config->embeddingsExtIdx < numExtInputs) {
            extInputs[config->embeddingsExtIdx] = decodeEmbedding;
        }

        // Attention mask
        if (config->maskExtIdx >= 0 && config->maskExtIdx < numExtInputs) {
            extInputs[config->maskExtIdx] = attentionMask;
        }

        // Position IDs
        if (config->posIdsExtIdx >= 0 && config->posIdsExtIdx < numExtInputs) {
            extInputs[config->posIdsExtIdx] = positionIds;
        }

        // Input IDs
        if (config->inputIdsExtIdx >= 0 && config->inputIdsExtIdx < numExtInputs) {
            extInputs[config->inputIdsExtIdx] = inputIds;
        }

        // Causal mask: wire into ext inputs (same pointer every step, updated in-place)
        if (causalMask != nullptr && config->causalMaskExtIdx >= 0 && config->causalMaskExtIdx < numExtInputs) {
            extInputs[config->causalMaskExtIdx] = causalMask;
        }

        // KV cache inputs: point to static buffers
        if (config->kvInputExtIndices != nullptr && staticKvBuffers != nullptr) {
            for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                int kvIdx = config->kvInputExtIndices[kv];
                if (kvIdx >= 0 && kvIdx < numExtInputs) {
                    extInputs[kvIdx] = staticKvBuffers[kv];
                }
            }
        }

        // ── Diagnostic: dump ext input fingerprints before plan execution ──
        // Always dump for first 2 steps (critical for debugging divergence at step 0/1)
        if (step <= 1) {
            NDArray* stepEmbed = extInputs[config->embeddingsExtIdx];
            if (stepEmbed != nullptr) {
                stepEmbed->syncToHost();
                float e0 = stepEmbed->e<float>(0);
                float e1 = stepEmbed->e<float>(1);
                float e2 = stepEmbed->e<float>(2);
                float e3 = stepEmbed->e<float>(3);
                sd_printf("DECODE_LOOP_DIAG step=%d PRE-EXECUTE embed[0..3]=[%f, %f, %f, %f] "
                         "embedShape=[%lld,%lld,%lld] embedPtr=%p\n",
                         step, e0, e1, e2, e3,
                         (long long)stepEmbed->sizeAt(0),
                         (long long)stepEmbed->sizeAt(1),
                         (long long)stepEmbed->sizeAt(2),
                         stepEmbed->specialBuffer());
            }
            if (config->posIdsExtIdx >= 0) {
                NDArray* posArr = extInputs[config->posIdsExtIdx];
                if (posArr != nullptr) {
                    posArr->syncToHost();
                    sd_printf("DECODE_LOOP_DIAG step=%d PRE-EXECUTE positionIds=%lld\n",
                             step, (long long)posArr->e<LongType>(0));
                }
            }
            if (config->inputIdsExtIdx >= 0) {
                NDArray* idsArr = extInputs[config->inputIdsExtIdx];
                if (idsArr != nullptr) {
                    idsArr->syncToHost();
                    sd_printf("DECODE_LOOP_DIAG step=%d PRE-EXECUTE inputIds=%lld\n",
                             step, (long long)idsArr->e<LongType>(0));
                }
            }
            if (config->maskExtIdx >= 0) {
                NDArray* maskArr = extInputs[config->maskExtIdx];
                if (maskArr != nullptr) {
                    maskArr->syncToHost();
                    sd_printf("DECODE_LOOP_DIAG step=%d PRE-EXECUTE attentionMask "
                             "shape=[%lld,%lld] first5=[",
                             step, (long long)maskArr->sizeAt(0), (long long)maskArr->sizeAt(1));
                    for (int mi = 0; mi < 5 && mi < maskArr->lengthOf(); mi++) {
                        sd_printf("%lld ", (long long)maskArr->e<LongType>(mi));
                    }
                    sd_printf("]\n");
                }
            }
            if (config->causalMaskExtIdx >= 0) {
                NDArray* cmaskArr = extInputs[config->causalMaskExtIdx];
                if (cmaskArr != nullptr) {
                    cmaskArr->syncToHost();
                    sd_printf("DECODE_LOOP_DIAG step=%d PRE-EXECUTE causalMask "
                             "shape=[%lld,%lld,%lld,%lld] first5=[",
                             step, (long long)cmaskArr->sizeAt(0), (long long)cmaskArr->sizeAt(1),
                             (long long)cmaskArr->sizeAt(2), (long long)cmaskArr->sizeAt(3));
                    for (int mi = 0; mi < 5 && mi < cmaskArr->lengthOf(); mi++) {
                        sd_printf("%f ", cmaskArr->e<float>(mi));
                    }
                    sd_printf("]\n");
                }
            }
            if (staticKvBuffers != nullptr && numKvPairs > 0) {
                NDArray* kv0 = staticKvBuffers[0];
                if (kv0 != nullptr) {
                    sd_printf("DECODE_LOOP_DIAG step=%d PRE-EXECUTE kv0 shape=[%lld,%lld,%lld,%lld] "
                             "ptr=%p\n",
                             step, (long long)kv0->sizeAt(0), (long long)kv0->sizeAt(1),
                             (long long)kv0->sizeAt(2), (long long)kv0->sizeAt(3),
                             kv0->specialBuffer());
                }
            }
            // Dump plan phase for debugging
            sd_printf("DECODE_LOOP_DIAG step=%d PRE-EXECUTE planPhase=%d frozen=%d\n",
                     step, (int)plan->getPlanPhase(),
                     plan->isShapesFrozen() ? 1 : 0);
        }

        // ── Step 2: Execute plan ──
        // Use executeSteadyState() for the hot path — it skips ~200ms of
        // per-step CPU overhead (lifecycle validation, buffer scanning,
        // fingerprinting, diagnostics). Falls back to full execute()
        // automatically if the plan hasn't reached steady state yet.
        Status planStatus = plan->executeSteadyState(
            extInputs, numExtInputs,
            planOutputs, numPlanOutputs,
            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));

        REQUIRE_TRUE(planStatus == Status::OK, 0,
                     "autoregressive_decode: plan execution FAILED at step %d with status %d. "
                     "Plan state: frozen=%d numExt=%d numOutputs=%d. "
                     "This is NOT recoverable — fix the plan execution failure.",
                     step, static_cast<int>(planStatus),
                     plan->isShapesFrozen() ? 1 : 0,
                     numExtInputs, numPlanOutputs);

        // Validate output was populated — hard error, not silent break.
        REQUIRE_TRUE(config->logitsOutputIdx < numPlanOutputs, 0,
                     "autoregressive_decode: logitsOutputIdx=%d >= numPlanOutputs=%d at step %d. "
                     "The plan has fewer outputs than expected.",
                     config->logitsOutputIdx, numPlanOutputs, step);
        REQUIRE_TRUE(planOutputs[config->logitsOutputIdx] != nullptr, 0,
                     "autoregressive_decode: logits output NDArray* is null at step %d (idx=%d). "
                     "Plan returned OK but did not populate the logits output slot.",
                     step, config->logitsOutputIdx);

        // Validate logits output buffer is not stale/closed.
        {
            NDArray* logitsArr = planOutputs[config->logitsOutputIdx];
            auto* logitsDb = logitsArr->dataBuffer();
            REQUIRE_TRUE(logitsDb != nullptr, 0,
                         "autoregressive_decode: logits DataBuffer is null at step %d. "
                         "Output array exists but has no backing buffer — likely a stale slot.",
                         step);
            REQUIRE_TRUE(!logitsDb->isClosed(), 0,
                         "autoregressive_decode: logits DataBuffer is CLOSED at step %d. "
                         "The plan reused a freed buffer — stale slot reuse bug.",
                         step);
            REQUIRE_TRUE(logitsArr->specialBuffer() != nullptr, 0,
                         "autoregressive_decode: logits specialBuffer (device ptr) is null at step %d. "
                         "Buffer exists but has no device allocation — missing syncToDevice or stale buffer.",
                         step);
        }

        // NOTE: Do NOT call plan->setShapesFrozen(true) here.
        // The plan auto-seals during its first execute() call, which sets
        // shapesFrozen=true and triggers Triton compilation. Calling
        // setShapesFrozen after execute violates the plan lifecycle
        // (executeCount > 0) and would skip the warmup/capture phase.
        // Auto-seal handles the transition correctly.

        // ── Step 3: Token sampling ──
        // Get logits from plan output at config->logitsOutputIdx
        NDArray* logitsOutput = planOutputs[config->logitsOutputIdx];

        // logitsOutput shape: [1, seqLen, vocabSize]
        // For decode steps (seqLen=1), it's [1, 1, vocabSize] — take [0, 0, :]
        // For prefill (seqLen=N), take [0, N-1, :] (last position)
        LongType logitsSeqLen = logitsOutput->sizeAt(1);
        LongType logitsVocab = logitsOutput->sizeAt(2);

        // Get pointer to last-position logits (already on device)
        NDArray::prepareSpecialUse({sampledToken}, {logitsOutput});

        if (temperature <= 0.0 || (topK <= 1 && topP <= 0.0)) {
            // Greedy: argmax over last-position logits
            // Compute offset to last position: (logitsSeqLen-1) * vocabSize
            LongType lastPosOffset = (logitsSeqLen - 1) * logitsVocab;
            const void* logitsPtr = static_cast<const char*>(logitsOutput->specialBuffer())
                                    + lastPosOffset * logitsOutput->sizeOfT();

            BUILD_SINGLE_SELECTOR(logitsOutput->dataType(), argmaxLauncher,
                                  (stream, logitsPtr, sampledToken->specialBuffer(), logitsVocab),
                                  SD_COMMON_TYPES);
        } else {
            // Sampling: use tokenSampleCuda with greedy fallback on logits pointer
            // For sampling, we pass the full logits output (tokenSampleCuda handles last-pos extraction)
            tokenSampleCuda(logitsOutput, sampledToken, temperature, topK, topP,
                            static_cast<LongType>(step), context);
        }

        NDArray::registerSpecialUse({sampledToken}, {logitsOutput});

        // Read sampled token ID back to host (single int64).
        // Issue D2H copy on the SAME stream as the argmax kernel to avoid
        // cross-stream sync. cudaMemcpyAsync orders after the argmax launch
        // on this stream (FIFO), then a single cudaStreamSynchronize waits
        // for both the kernel and the copy to complete.
        // This replaces: cudaStreamSynchronize(*stream) + e<>() which did
        // TWO full pipeline drains (one on exec stream, one on stream 0).
        LongType nextTokenId = 0;
        cudaMemcpyAsync(&nextTokenId, sampledToken->specialBuffer(),
                        sizeof(LongType), cudaMemcpyDeviceToHost, *stream);
        cudaStreamSynchronize(*stream);

        // ── Diagnostic: dump logits fingerprint for determinism debugging ──
        // Always dump for first 2 steps (critical for debugging divergence)
        if (step <= 1) {
            logitsOutput->syncToHost();
            float l0 = logitsOutput->e<float>(0, logitsSeqLen-1, 0);
            float l1 = logitsOutput->e<float>(0, logitsSeqLen-1, 1);
            float l2 = logitsOutput->e<float>(0, logitsSeqLen-1, 2);
            float l3 = logitsOutput->e<float>(0, logitsSeqLen-1, 3);
            sd_printf("DECODE_LOOP_DIAG step=%d POST-EXECUTE token=%lld logits[0..3]=[%f, %f, %f, %f] "
                     "vocabSize=%lld seqLen=%lld\n",
                     step, (long long)nextTokenId, l0, l1, l2, l3,
                     (long long)logitsVocab, (long long)logitsSeqLen);
            // Dump top-5 logits with indices for deeper comparison
            int topKDump = 5;
            if (logitsVocab < topKDump) topKDump = (int)logitsVocab;
            struct { float val; int idx; } top5[5];
            for (int tk = 0; tk < topKDump; tk++) {
                top5[tk].val = logitsOutput->e<float>(0, logitsSeqLen-1, tk);
                top5[tk].idx = tk;
            }
            for (int vi = topKDump; vi < logitsVocab; vi++) {
                float v = logitsOutput->e<float>(0, logitsSeqLen-1, vi);
                int minIdx = 0;
                for (int tk = 1; tk < topKDump; tk++) {
                    if (top5[tk].val < top5[minIdx].val) minIdx = tk;
                }
                if (v > top5[minIdx].val) {
                    top5[minIdx].val = v;
                    top5[minIdx].idx = vi;
                }
            }
            // Simple bubble sort for display
            for (int i = 0; i < topKDump-1; i++) {
                for (int j = i+1; j < topKDump; j++) {
                    if (top5[j].val > top5[i].val) {
                        auto tmp = top5[i];
                        top5[i] = top5[j];
                        top5[j] = tmp;
                    }
                }
            }
            sd_printf("DECODE_LOOP_DIAG step=%d POST-EXECUTE top5_logits=[", step);
            for (int tk = 0; tk < topKDump; tk++) {
                sd_printf("(idx=%d,val=%f) ", top5[tk].idx, top5[tk].val);
            }
            sd_printf("]\n");
        }

        // Store in output
        generatedTokenIds->p(tokensGenerated, nextTokenId);
        tokensGenerated++;

        // ── Step 4: Check stop condition ──
        bool shouldStop = false;
        for (int s : stopTokenIds) {
            if (nextTokenId == static_cast<LongType>(s)) {
                shouldStop = true;
                break;
            }
        }

        auto stepEnd = std::chrono::high_resolution_clock::now();
        double stepMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
        stepTimesMs.push_back(stepMs);

        if (shouldStop) break;

        // ── Step 5: KV scatter — copy present KV into static buffers ──
        // Skip manual scatter when the plan's native KV scatter is active
        // (planOwnsKvScatter) — executeKvScatterPostExec handles it with its
        // own device-side position counter via executeSteadyState.
        if (!config->planOwnsKvScatter &&
            config->kvOutputIndices != nullptr && staticKvBuffers != nullptr && numKvPairs > 0) {
            // Build batched KV scatter entries — validate every buffer first.
            std::vector<KvScatterEntry> entries(2 * numKvPairs);
            for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                int kvOutIdx = config->kvOutputIndices[kv];
                REQUIRE_TRUE(kvOutIdx >= 0 && kvOutIdx < numPlanOutputs, 0,
                             "autoregressive_decode: KV output index %d out of range [0,%d) "
                             "at step %d kv=%d",
                             kvOutIdx, numPlanOutputs, step, kv);
                NDArray* presentKv = planOutputs[kvOutIdx];
                NDArray* staticBuf = staticKvBuffers[kv];
                REQUIRE_TRUE(presentKv != nullptr, 0,
                             "autoregressive_decode: KV output[%d] (planOutput[%d]) is null "
                             "at step %d — plan did not produce this output.",
                             kv, kvOutIdx, step);
                REQUIRE_TRUE(staticBuf != nullptr, 0,
                             "autoregressive_decode: static KV buffer[%d] is null at step %d.",
                             kv, step);
                REQUIRE_TRUE(presentKv->specialBuffer() != nullptr, 0,
                             "autoregressive_decode: KV output[%d] has null device buffer "
                             "at step %d — stale or uninitialized output.",
                             kv, step);
                REQUIRE_TRUE(staticBuf->specialBuffer() != nullptr, 0,
                             "autoregressive_decode: static KV[%d] has null device buffer "
                             "at step %d — buffer was freed or never allocated.",
                             kv, step);

                entries[kv].srcPtr = presentKv->specialBuffer();
                entries[kv].dstPtr = staticBuf->specialBuffer();
                entries[kv].heads = presentKv->sizeAt(1);
                entries[kv].srcSeqLen = presentKv->sizeAt(2);
                entries[kv].dstSeqLen = staticBuf->sizeAt(2);
                entries[kv].dim = presentKv->sizeAt(3);
                entries[kv].lastPos = presentKv->sizeAt(2) - 1;
                entries[kv].cachePos = currentPosition;
            }

            kvScatterBatched(entries.data(), 2 * numKvPairs,
                             staticKvBuffers[0]->dataType(), context);

            // Mark static KV buffers as device-authoritative after scatter.
            // Without this, isPrimaryActual() remains true (from Java's initial
            // .assign()/.putScalar() setup), causing the frozen fast path to
            // force H2D every step — overwriting valid device KV data with
            // stale host zeros → degenerate output + 56 synchronous H2D copies
            // per step (~100ms overhead).
            for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                staticKvBuffers[kv]->tickWriteDevice();
            }
        }

        // ── Step 6: Embedding lookup for next token ──
        NDArray::prepareSpecialUse({decodeEmbedding}, {embeddingTable});
        BUILD_SINGLE_SELECTOR(embeddingTable->dataType(), embedLookupLauncher,
                              (stream, embeddingTable->specialBuffer(),
                               decodeEmbedding->specialBuffer(),
                               nextTokenId, hidden, embTableRowStride),
                              SD_COMMON_TYPES);
        NDArray::registerSpecialUse({decodeEmbedding}, {embeddingTable});

        // ── Advance position BEFORE updating mask/posIds for the next step ──
        // KV scatter (step 5) used currentPosition as the cache write position.
        // Now increment so the mask/posIds updates below prepare the NEXT decode
        // step's inputs (next KV position to attend to, next position_ids value).
        currentPosition++;

        // ── Step 7: Update input buffers for next step ──
        // Update attention mask: unmask the KV position that was JUST written.
        // currentPosition was incremented above; currentPosition - 1 is the last
        // written KV slot.  This matches the Java DecoderInputBuilder delta-update
        // pattern: mask.putScalar(0, cachePos - 1) = 1.
        LongType kvJustWritten = currentPosition - 1;
        if (kvJustWritten >= 0 && kvJustWritten < maxKvLen) {
            NDArray::prepareSpecialUse({attentionMask}, {});
            BUILD_SINGLE_SELECTOR(attentionMask->dataType(), updateAttentionMaskLauncher,
                                  (stream, attentionMask->specialBuffer(), kvJustWritten, maxKvLen),
                                  SD_COMMON_TYPES);
            NDArray::registerSpecialUse({attentionMask}, {});
        }

        // Update causal mask: unmask the new position (set to 0.0f)
        // Mirrors Java buildCausalMask delta update: putScalar(cachePos, 0.0f)
        if (causalMask != nullptr && currentPosition < causalMaskLen) {
            NDArray::prepareSpecialUse({causalMask}, {});
            BUILD_SINGLE_SELECTOR(causalMask->dataType(), updateCausalMaskLauncher,
                                  (stream, causalMask->specialBuffer(), currentPosition, causalMaskLen),
                                  SD_FLOAT_TYPES);
            NDArray::registerSpecialUse({causalMask}, {});
        }

        // Update attn_mask_reformat: unmask the just-written KV position (set to 0.0f)
        // Mirrors Java DecoderInputBuilder delta update: bias[0..cachePos) = 0.0f
        if (attnMaskReformat != nullptr && kvJustWritten >= 0 && kvJustWritten < attnMaskReformatLen) {
            NDArray::prepareSpecialUse({attnMaskReformat}, {});
            BUILD_SINGLE_SELECTOR(attnMaskReformat->dataType(), updateCausalMaskLauncher,
                                  (stream, attnMaskReformat->specialBuffer(), kvJustWritten, attnMaskReformatLen),
                                  SD_FLOAT_TYPES);
            NDArray::registerSpecialUse({attnMaskReformat}, {});
        }

        // Update position_ids: set to next step's position
        NDArray::prepareSpecialUse({positionIds}, {});
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            positionIds->specialBuffer(),
            currentPosition);
        NDArray::registerSpecialUse({positionIds}, {});

        // Update input_ids: set to next token
        NDArray::prepareSpecialUse({inputIds}, {});
        updateInputIdsKernel<<<1, 1, 0, *stream>>>(
            inputIds->specialBuffer(),
            nextTokenId);
        NDArray::registerSpecialUse({inputIds}, {});
    }

    // ── Final sync ──
    cudaStreamSynchronize(*stream);

    // ── Write token count ──
    tokenCount->p(0, static_cast<LongType>(tokensGenerated));

    // ── Compute timing stats ──
    auto loopEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(loopEnd - loopStart).count();

    if (!stepTimesMs.empty()) {
        double avgMs = totalMs / stepTimesMs.size();
        double tokPerSec = stepTimesMs.size() > 0 ? (stepTimesMs.size() * 1000.0 / totalMs) : 0.0;

        std::vector<double> sorted = stepTimesMs;
        std::sort(sorted.begin(), sorted.end());
        double p50 = sorted[sorted.size() / 2];
        double p99 = sorted[std::min<size_t>(sorted.size() - 1,
                                              static_cast<size_t>(sorted.size() * 0.99))];

        timingInfo->p(0, static_cast<float>(totalMs));
        timingInfo->p(1, static_cast<float>(avgMs));
        timingInfo->p(2, static_cast<float>(tokPerSec));
        timingInfo->p(3, static_cast<float>(p50));
        timingInfo->p(4, static_cast<float>(p99));
    }

    // ── Cleanup internal allocations ──
    // decodeEmbedding is NOT deleted — it's prefillEmbeddings, owned by the caller.
    delete sampledToken;
    if (internalMask != nullptr) {
        delete internalMask;
    }
    if (internalPosIds != nullptr) {
        delete internalPosIds;
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
