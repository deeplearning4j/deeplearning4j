/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_HELPERS_AUTOREGRESSIVE_DECODE_H
#define LIBND4J_HELPERS_AUTOREGRESSIVE_DECODE_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>
#include <ops/declarable/helpers/token_sample.h>

#include <vector>

namespace sd {

// Forward declaration — full definition in NativeDynamicShapePlan.h
namespace graph {
class NativeDynamicShapePlan;
}

namespace ops {
namespace helpers {

/**
 * Host-side token notification emitted by the canonical native decode loop.
 * The callback runs synchronously after a token has been committed to the
 * generated-token output. It must not retain pointers owned by the decoder.
 */
using AutoregressiveTokenCallback = void (*)(LongType tokenId, void* userData);

/**
 * Cooperative cancellation probe. The decoder polls it between complete
 * decode steps, preserving KV/recurrent state at a resumable boundary.
 */
using AutoregressiveCancelCallback = bool (*)(void* userData);

/**
 * Configuration for the autoregressive decode loop.
 * Passed from the op to the platform-specific helper.
 */
struct AutoregressiveDecodeConfig {
    // Plan execution
    graph::NativeDynamicShapePlan* planHandle;  // compiled decoder plan (nullable for legacy path)
    NDArray** planExternalInputs;               // full ext input array for plan (owned by caller)
    int numPlanExternalInputs;
    NDArray** planOutputs;                      // pre-allocated output pointer array
    int numPlanOutputs;
    void* extInputContext;                      // OpaqueContext* with ext inputs registered (persistent)

    // Mutable input indices in the plan's external input array
    int embeddingsExtIdx;       // inputs_embeds
    int maskExtIdx;             // attention_mask
    int causalMaskExtIdx;       // _causal_mask (-1 if not used)
    int posIdsExtIdx;           // position_ids
    int inputIdsExtIdx;         // input_ids
    int logitsOutputIdx;        // which plan output is logits
    int attnMaskReformatExtIdx = -1; // attn_mask_reformat ext input (-1 if not used)

    // KV cache indices
    int* kvInputExtIndices;     // ext input indices for past_key_values (2*numKvPairs)
    int* kvOutputIndices;       // plan output indices for present KVs (2*numKvPairs)

    // In-graph KV cache (GGUF pattern): the attention op writes K/V in-place
    // at cachePosition. These ext input indices point to the scalar position
    // tensors updated per step. -1 means not used (ONNX/non-GGUF path).
    int positionOffsetExtIdx = -1;      // position_offset scalar (RoPE position)
    int cachePositionExtIdx = -1;       // cache_position scalar (KV write position)
    int actualSequenceLengthExtIdx = -1; // actual_sequence_length scalar (live recurrent timesteps)

    // KV scatter ownership: when true, the plan's native KV scatter
    // (configureKvScatter / executeKvScatterPostExec) handles KV cache
    // updates. The decode loop skips its own manual scatter to avoid
    // double-writing with independent position counters.
    bool planOwnsKvScatter = false;

    // GDN (Gated Delta Net) recurrent state feedback.
    // For hybrid architectures (e.g. Qwen3-0.6B with 18 GDN + 6 attention layers),
    // the GDN layers have recurrent state that must be copied from outputs back to
    // inputs between decode steps. Without this, GDN layers see frozen state from
    // the warmup step and the model degenerates.
    int* gdnStateExtIndices = nullptr;     // ext input indices for past_gdn_state.{layer}
    int* gdnStateOutputIndices = nullptr;  // plan output indices for gdn_state_out_{layer}
    int numGdnStatePairs = 0;

    // Conv state feedback (1D conv state in GDN layers).
    int* convStateExtIndices = nullptr;    // ext input indices for past_conv_state.{layer}
    int* convStateOutputIndices = nullptr; // plan output indices for conv_state_out_{layer}
    int numConvStatePairs = 0;

    // ─── ADR 0107 V2: Quantised KV cache side-channel ────────────────────────
    //
    // When kvQuantFormat > 0 (INT8_KV mode), the KV buffers in planExternalInputs
    // (at indices kvInputExtIndices[i]) are INT8 arrays. The corresponding float
    // per-token-per-head scale arrays are passed here as a side channel — they are
    // NOT registered as plan ext inputs (the SameDiff model graph has no scale vars),
    // but are passed directly to the attention helper (kvInPlaceWriteQuantisedBSHD /
    // fusedGQADecodeQuantisedCuda) by the native decode op.
    //
    // Layout: kvScaleBuffers[0..numKvPairs-1] = key scales per layer
    //         kvScaleBuffers[numKvPairs..2*numKvPairs-1] = value scales per layer
    // Shape of each: [batch, maxKvLen, kvHeads] float32.
    // Null when kvQuantFormat == 0 (standard float KV path).
    NDArray** kvScaleBuffers = nullptr;    // 2*numKvPairs scale arrays (null = float KV)
    int kvQuantFormat = 0;                 // 0=float, 1=INT8_KV, 5=FP16K_INT8V

    // ─── ADR 0106 Phase 1: fixed W_max window substrate ──────────────────────
    //
    // When activeWindow > 1, the per-step forward runs over a fixed [1,1,W_max,past+W_max]
    // masked attention window and a fixed [1,W_max] position grid. Both tensors are
    // pre-allocated at W_max size and never reallocated — device addresses stay stable
    // for CUDA-graph capture (ADR 0105 pointer-stability contract). Positions are
    // activated via mask bits, not by reallocation or reshape.
    //
    // When activeWindow == 1 (the default), windowGridMask and windowPositionGrid are
    // nullptr and the existing single-token path runs unchanged (bit-identical).
    //
    // Layout of windowGridMask: [1, 1, W_max, past + W_max] FLOAT
    //   Row w (query position w): mask[past_len + w] = 0 (attend to self)
    //                             mask[past_len + j] = MASK_FILL for j != w  (causal)
    //                             mask[0..past_len-1] = 0  (attend to all past KV)
    //   Inactive rows (w >= activeWindow): mask[*] = MASK_FILL (all masked)
    //
    // Layout of windowPositionGrid: [1, W_max] INT64
    //   grid[w] = currentPosition + w  for w < activeWindow
    //   grid[w] = currentPosition      for w >= activeWindow (doesn't matter; masked)
    //
    // The decode loop selects the token at position 0 (the first window slot) for
    // the next input — exactly the greedy W=1 behaviour. Phase 2 (speculative) will
    // inspect all W position-logits and apply its accept/rollback policy.

    NDArray* windowGridMask = nullptr;       // [1,1,W_max,past+W_max] FLOAT — null when W=1
    NDArray* windowPositionGrid = nullptr;   // [1,W_max] INT64           — null when W=1
    int windowMax = 1;                       // maximum number of candidate window positions
    int activeWindow = 1;                    // active positions this step (<=windowMax)

    // ─── ADR 0106 Phase 2: n-gram speculative decoding ────────────────────────
    //
    // When speculativeK > 0 and windowMax >= speculativeK+1, the decode loop
    // proposes up to speculativeK draft tokens per step using a bigram (n-gram)
    // proposer. The forward pass runs over activeWindow = 1+proposed positions
    // (filling only those slots in the W_max mask). All accepted tokens are
    // emitted at once — the lossless accept rule guarantees token-by-token
    // greedy equivalence.
    //
    // speculativeK == 0 (default) disables speculation — the W=1 path runs
    // completely unchanged (bit-identical to ADR 0106 Phase 1).
    //
    // speculatorType: 0=none, 1=NGRAM, 2=Qwen3.5 bundled MTP predictor.
    int speculativeK = 0;            // max draft tokens per step (0 = off)
    int speculatorType = 0;          // 0=none, 1=NGRAM, 2=MTP

    // ─── Qwen3.5 bundled MTP predictor ─────────────────────────────────────────
    // The predictor is a second plan over the same immutable SameDiff weights. It owns an
    // independent context and KV cache, and always executes scalar [1,1] steps. The target plan
    // remains W-wide and verifies all proposed tokens in one replay.
    graph::NativeDynamicShapePlan* mtpPlanHandle = nullptr;
    void* mtpExtInputContext = nullptr;
    int mtpNumPlanExternalInputs = 0;
    int mtpNumPlanOutputs = 0;
    int mtpInputIdsExtIdx = -1;
    int mtpTargetHiddenExtIdx = -1;
    int mtpCausalMaskExtIdx = -1;
    int mtpPositionOffsetExtIdx = -1;
    int mtpCachePositionExtIdx = -1;
    int mtpKvInputExtIndices[2] = {-1, -1};
    int mtpLogitsOutputIdx = -1;
    int mtpHiddenOutputIdx = -1;
    int targetHiddenOutputIdx = -1;  // pre-final-norm target hidden rows

    // Stable-address arrays also passed as op inputs to retain lifetime and make native updates
    // explicit. The same NDArray objects are registered in mtpExtInputContext.
    NDArray* mtpInputIds = nullptr;
    NDArray* mtpTargetHidden = nullptr;
    NDArray* mtpCausalMask = nullptr;
    NDArray* mtpPositionOffset = nullptr;
    NDArray* mtpCachePosition = nullptr;
    NDArray* mtpKvBuffers[2] = {nullptr, nullptr};

    // Unified token-selection policy. Scalar greedy/sample are supported today; wider policies are
    // parsed and validated by tokenSamplePolicy so they cannot silently run as greedy.
    TokenSampleConfig sampleConfig;

    // Portable streaming/cancellation hooks used by SDX generation sessions.
    // tokenCallback is notification-only. cancelCallback is polled between
    // complete decode steps so a cancelled session retains coherent KV state.
    AutoregressiveTokenCallback tokenCallback = nullptr;
    AutoregressiveCancelCallback cancelCallback = nullptr;
    void* callbackUserData = nullptr;
};

/**
 * Autoregressive decode loop (CPU and CUDA — platform selected at link time).
 *
 * When config->planHandle is non-null, executes the full native decode loop:
 *   plan.execute() → token_sample → kv_scatter → embed_lookup → input update → repeat
 *
 * When config is null or planHandle is null, falls back to the legacy path
 * (mask/pos update only — plan execution done by Java caller).
 */
SD_LIB_HIDDEN void autoregressiveDecode(
    NDArray* prefillEmbeddings,    // [1, seqLen, hidden]
    NDArray* embeddingTable,       // [vocabSize, hidden]
    NDArray* inputIds,             // [1, seqLen] INT64
    NDArray* attentionMask,        // [1, 1, seqLen, maxKvLen] (or null → built internally)
    NDArray* positionIds,          // [1, seqLen] INT64 (or null → built internally)
    NDArray** staticKvBuffers,     // 2*numKvPairs buffers (or null)
    int numKvPairs,
    NDArray* generatedTokenIds,    // [maxNewTokens] INT64 output
    NDArray* tokenCount,           // [1] INT64 output
    NDArray* timingInfo,           // [10] FLOAT output
    int maxNewTokens,
    int prefillSeqLen,
    const std::vector<int>& stopTokenIds,
    double temperature,
    int topK,
    double topP,
    double repPenalty,
    LaunchContext* context,
    AutoregressiveDecodeConfig* config = nullptr);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_AUTOREGRESSIVE_DECODE_H
