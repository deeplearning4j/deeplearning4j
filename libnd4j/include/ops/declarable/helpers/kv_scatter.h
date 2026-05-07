/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_HELPERS_KV_SCATTER_H
#define LIBND4J_HELPERS_KV_SCATTER_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Copy present[batch, heads, lastPos, dim] -> output[batch, heads, cachePos, dim]
 *
 * @param present   [batch, heads, seqLen, dim] source tensor
 * @param output    [batch, heads, maxKvLen, dim] destination tensor (modified in-place)
 * @param cachePos  position along dim 2 in output to write to
 * @param context   launch context
 */
SD_LIB_HIDDEN void kvScatter(NDArray* present, NDArray* output,
                              LongType cachePos, LaunchContext* context);

/**
 * Descriptor for one KV scatter entry in the batched kernel.
 */
struct KvScatterEntry {
    const void* srcPtr;   // present's specialBuffer
    void* dstPtr;         // static buffer's specialBuffer
    LongType heads;
    LongType srcSeqLen;
    LongType dstSeqLen;
    LongType dim;
    LongType lastPos;     // srcSeqLen - 1
    LongType cachePos;
};

/**
 * Batch multiple KV scatter operations into a single kernel launch.
 * Eliminates per-mapping kernel launch overhead (60 launches → 1).
 *
 * @param entries    array of scatter descriptors
 * @param numEntries number of entries
 * @param dtype      data type of all entries (must be uniform)
 * @param context    launch context
 */
SD_LIB_HIDDEN void kvScatterBatched(const KvScatterEntry* entries, int numEntries,
                                      DataType dtype, LaunchContext* context);

/**
 * Descriptor for a dynamic-position KV scatter entry.
 * Like KvScatterEntry but cachePos is read from a device-side (or host-side on CPU)
 * int64 scalar at execution time — enabling CUDA graph replay where only the
 * position value changes between steps (the pointer stays fixed).
 *
 * kvPosPtr must point to a long/int64 value containing the current cache position.
 * On CUDA: this MUST be a device-accessible address (allocated via cudaMalloc/pool).
 * On CPU:  this can be any int64* (the value is read directly).
 */
struct KvScatterDynEntry {
    const void* srcPtr;    // present's specialBuffer (device) / buffer (CPU)
    void* dstPtr;          // static buffer's specialBuffer (device) / buffer (CPU)
    const LongType* kvPosPtr;  // pointer to int64 cache position (device on CUDA, host on CPU)
    LongType heads;
    LongType srcSeqLen;
    LongType dstSeqLen;
    LongType dim;
    LongType lastPos;      // srcSeqLen - 1
};

/**
 * Batch multiple dynamic-position KV scatter operations.
 *
 * Like kvScatterBatched but reads cachePos from entries[i].kvPosPtr at runtime.
 * This is CUDA graph compatible: the position pointer address is baked into the
 * graph at capture time; only the VALUE at that address changes between replays
 * (via cudaMemcpyAsync before each replay).
 *
 * All entries MUST share the same kvPosPtr (they all use the same cache position).
 *
 * @param entries    array of dynamic scatter descriptors
 * @param numEntries number of entries
 * @param dtype      data type of all entries (must be uniform)
 * @param context    launch context
 */
SD_LIB_HIDDEN void kvScatterDynBatched(const KvScatterDynEntry* entries, int numEntries,
                                        DataType dtype, LaunchContext* context);

/**
 * Write new K/V data at cache_position into the past KV buffer in-place.
 *
 * Cache position is read from a device-side pointer (CUDA) or host pointer (CPU),
 * making this CUDA graph compatible: the pointer address is baked in at capture
 * time; the value changes between replays.
 *
 * Layout: pastKv is BHSD [batch, numKvHeads, maxSeqLen, headDim].
 *         newKv is BSHD [batch, seqKV, numKvHeads, headDim] after reshape.
 *         Typically seqKV=1 during decode.
 *
 * The kernel writes newKv[b, s, h, d] → pastKv[b, h, cachePos+s, d].
 *
 * @param pastKv      [batch, numKvHeads, maxSeqLen, headDim] BHSD — modified in-place
 * @param newKv       [batch, seqKV, numKvHeads, headDim] BSHD — source data
 * @param cachePosPtr pointer to int64 cache position (device on CUDA, host on CPU)
 * @param context     launch context
 */
SD_LIB_HIDDEN void kvInPlaceWrite(NDArray* pastKv, NDArray* newKv,
                                   const void* cachePosPtr, LaunchContext* context);

/**
 * Write new K/V data at cache_position into a BSHD-layout KV cache in-place.
 *
 * Unlike kvInPlaceWrite (BHSD pastKv, BSHD newKv), both arrays here are BSHD.
 * This is used by dot_product_attention_v2 where the KV cache is BSHD.
 *
 * Rank-4 BSHD: cache [batch, maxSeqLen, numKvHeads, headDim]
 *              newKv  [batch, seqKV, numKvHeads, headDim]
 *              Writes newKv[b, s, h, d] → cache[b, cachePos+s, h, d].
 *
 * Rank-3 BSF:  cache [batch, maxSeqLen, features]
 *              newKv  [batch, seqKV, features]
 *              Writes newKv[b, s, f] → cache[b, cachePos+s, f].
 *
 * Cache position is read from a device-side pointer (CUDA) or host pointer (CPU),
 * making this CUDA graph compatible.
 *
 * @param cache       BSHD [batch, maxSeqLen, heads, dim] or BSF [batch, maxSeqLen, features] — modified in-place
 * @param newKv       BSHD [batch, seqKV, heads, dim] or BSF [batch, seqKV, features] — source data
 * @param cachePosPtr pointer to int64 cache position (device on CUDA, host on CPU)
 * @param context     launch context
 */
SD_LIB_HIDDEN void kvInPlaceWriteBSHD(NDArray* cache, NDArray* newKv,
                                       const void* cachePosPtr, LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
