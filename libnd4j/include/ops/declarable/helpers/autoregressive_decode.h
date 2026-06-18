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

#include <vector>

namespace sd {

// Forward declaration — full definition in NativeDynamicShapePlan.h
namespace graph {
class NativeDynamicShapePlan;
}

namespace ops {
namespace helpers {

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
    int positionOffsetExtIdx = -1;  // position_offset scalar (RoPE position)
    int cachePositionExtIdx = -1;   // cache_position scalar (KV write position)

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
    NDArray* timingInfo,           // [5] FLOAT output
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
