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

#ifndef LIBND4J_KV_CACHE_QUANTIZE_H
#define LIBND4J_KV_CACHE_QUANTIZE_H

#include <array/NDArray.h>
#include <execution/LaunchContext.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

// Quantization format enum
enum class KVQuantFormat : int {
    INT8 = 0,
    FP8_E4M3 = 1,
    FP8_E5M2 = 2,
    INT4 = 3
};

SD_LIB_HIDDEN void kvCacheQuantize(
    NDArray* input,       // float KV data [...]
    NDArray* quantized,   // output quantized data [...]
    NDArray* scales,      // output per-channel scales [...]
    int quantFormat,      // KVQuantFormat
    LaunchContext* context = nullptr);

SD_LIB_HIDDEN void kvCacheDequantize(
    NDArray* quantized,   // quantized data [...]
    NDArray* scales,      // per-channel scales [...]
    NDArray* output,      // float output [...]
    int quantFormat,
    LaunchContext* context = nullptr);

/**
 * V2 quantised-on-write helper: append one decode-step K or V vector into a fixed-allocation
 * INT8 cache buffer and write the per-token-per-head scale.
 *
 * Called once per attention layer per decode step (inside dot_product_attention_v2 on the
 * QUANTIZED path). CUDA-graph safe: all pointer addresses are stable; the scatter position is
 * read from device-side cachePosPtr at kernel runtime (same pattern as kvInPlaceWriteBSHD).
 *
 * Layout contract (INT8_KV mode):
 *   newKv      : float  [batch, 1,        kvHeads, headDim]  — current-step K or V
 *   quantCache : INT8   [batch, maxKvLen,  kvHeads, headDim]  — fixed pre-allocated cache
 *   scaleCache : float  [batch, maxKvLen,  kvHeads]           — per-token-per-head scale
 *   cachePosPtr: device-resident int64 scalar — the write position
 *
 * Per-row (per KV head) absmax reduction + INT8 scatter + scale scatter.
 * No cross-row atomics → capture-safe (warp reduction within each head row).
 */
SD_LIB_HIDDEN void kvInPlaceWriteQuantisedBSHD(
    NDArray* quantCache,     // INT8 [batch, maxKvLen, kvHeads, headDim] — modified in-place
    NDArray* scaleCache,     // float [batch, maxKvLen, kvHeads]          — modified in-place
    NDArray* newKv,          // float [batch, 1, kvHeads, headDim]        — source (current step)
    const void* cachePosPtr, // pointer to int64 write position
    LaunchContext* context);

/**
 * ADR 0107 V2 — Thread-local KV scale buffer registry.
 *
 * The native decode loop calls setKvScaleRegistry() before each plan execution to inject
 * the per-layer scale arrays. dot_product_attention_v2 calls lookupKvScaleByCache() to
 * retrieve the scale arrays for the INT8 KV cache pointer it received.
 *
 * This avoids adding scale variables to the SameDiff model graph and avoids extending the
 * frozen plan's ext input array — scale arrays are passed as a side channel that the decode
 * loop owns and the attention op queries per execution.
 *
 * Thread-safety: each decode thread has its own registry (thread_local).
 * Capture-safety: the registry holds raw NDArray* — stable device pointers, no allocation.
 *
 * Layout contract:
 *   kvQuantBuffers[0..numKvPairs-1]  = INT8 key caches per layer (key)
 *   kvScaleBuffers[0..numKvPairs-1]  = float key scales per layer (value for key lookup)
 *   kvScaleBuffers[numKvPairs..2N-1] = float value scales per layer (value for val lookup)
 * Each value cache uses the same layer index: valScale = kvScaleBuffers[numKvPairs + layerIdx].
 */
// Registry state + accessors are defined INLINE here (not in a backend-specific .cpp)
// so the definitions link into BOTH the CPU and CUDA .so. The callers span backends —
// autoregressive_decode.cu (CUDA link) and dot_product_attention_v2.cpp (both) — so a
// definition living only in helpers/cpu/kv_cache_quantize.cpp left the CUDA link with
// undefined references. `inline` (+ `inline thread_local`, C++17) gives one merged
// instance across all translation units.
struct KvScaleRegistry {
    NDArray** kvQuantBuffers = nullptr;  // INT8 key caches (lookup keys)
    NDArray** kvScaleBuffers = nullptr;  // float scales [0..N-1]=key, [N..2N-1]=val
    int numKvPairs = 0;
};

inline thread_local KvScaleRegistry tl_kvScaleRegistry;

inline void setKvScaleRegistry(
    NDArray** kvQuantBuffers,     // INT8 key cache pointers [numKvPairs] — used as lookup keys
    NDArray** kvScaleBuffers,     // float scale arrays [2*numKvPairs] (key then value)
    int numKvPairs) {
    tl_kvScaleRegistry.kvQuantBuffers = kvQuantBuffers;
    tl_kvScaleRegistry.kvScaleBuffers = kvScaleBuffers;
    tl_kvScaleRegistry.numKvPairs = numKvPairs;
}

inline void clearKvScaleRegistry() {
    tl_kvScaleRegistry.kvQuantBuffers = nullptr;
    tl_kvScaleRegistry.kvScaleBuffers = nullptr;
    tl_kvScaleRegistry.numKvPairs = 0;
}

/**
 * Look up key and value scale arrays by INT8 cache pointer identity.
 * Called from dot_product_attention_v2 when keyCache->dataType() == INT8
 * and keyScaleCache is null (V2 mode but without inputs 9/10 in the graph).
 * Returns false if the registry is empty or the cache pointer is not found.
 */
inline bool lookupKvScaleByCache(
    const NDArray* kvCache,       // INT8 key cache pointer (used as key)
    NDArray** outKeyScale,        // out: float key scale array or nullptr
    NDArray** outValScale) {      // out: float value scale array or nullptr
    const KvScaleRegistry& reg = tl_kvScaleRegistry;
    if (reg.kvQuantBuffers == nullptr || reg.numKvPairs == 0) return false;
    for (int i = 0; i < reg.numKvPairs; i++) {
        if (reg.kvQuantBuffers[i] == kvCache) {
            *outKeyScale = reg.kvScaleBuffers[i];
            *outValScale = reg.kvScaleBuffers[reg.numKvPairs + i];
            return true;
        }
    }
    return false;
}

/**
 * CPU reference dequantised GQA decode attention.
 * Equivalent to fusedGQADecodeCuda but for CPU: loads INT8 K/V, applies per-token-per-head
 * scale, then runs standard Q@K^T + softmax + attn@V. Used for correctness testing.
 *
 * quantKeyCache  : INT8  [batch, seqKV, kvHeads, headDim]
 * keyScaleCache  : float [batch, seqKV, kvHeads]
 * quantValCache  : INT8  [batch, seqKV, kvHeads, headDim]
 * valScaleCache  : float [batch, seqKV, kvHeads]
 * query          : float [batch, 1,     qHeads,  headDim]
 * output         : float [batch, 1,     qHeads,  headDim]
 * attentionBias  : float [batch, 1/qHeads, 1, seqKV] or nullptr
 */
SD_LIB_HIDDEN void fusedGQADecodeQuantisedCpu(
    NDArray* query,
    NDArray* quantKeyCache,
    NDArray* keyScaleCache,
    NDArray* quantValCache,
    NDArray* valScaleCache,
    NDArray* output,
    double scale,
    NDArray* attentionBias,
    LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
