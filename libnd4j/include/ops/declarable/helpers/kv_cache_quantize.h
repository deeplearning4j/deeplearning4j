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
#include <math/templatemath.h>
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

// kvCacheQuantize/kvCacheDequantize quantize rows that run along the LAST dimension. Every operand
// is addressed through its own shape and strides (any order, any view): row r is TAD r along the
// last dimension, TADs being enumerated in C order over the leading dimensions, and its FLOAT32
// scale is the C-order element r of the scales array.
//   INT8/FP8: quantized has the input's shape, one int8 per element.
//   INT4:     quantized has the input's shape; element pair (2j, 2j+1) of a row packs into byte j
//             of the same row (low nibble = even element, value + 8), and bytes
//             [ceil(rowLen/2), rowLen) are zero. Every row starts at its own logical row, so any
//             slice of the tensor along the leading dimensions is itself a valid INT4 tensor.
//   ROW-INLINE (scales == nullptr, INT8/FP8 only): quantized is [leading..., rowLen + 4] with a
//             unit last-dimension stride; bytes [rowLen, rowLen + 4) of each row hold its scale.
SD_LIB_HIDDEN void kvCacheQuantize(
    NDArray* input,       // float KV data [..., rowLen]
    NDArray* quantized,   // output quantized data [..., rowLen] or [..., rowLen + 4] row-inline
    NDArray* scales,      // output FLOAT32 per-row scales, numRows elements; nullptr = row-inline
    int quantFormat,      // KVQuantFormat
    LaunchContext* context = nullptr);

SD_LIB_HIDDEN void kvCacheDequantize(
    NDArray* quantized,   // quantized data, same shape as output
    NDArray* scales,      // FLOAT32 per-row scales, numRows elements
    NDArray* output,      // float output [..., rowLen]
    int quantFormat,
    LaunchContext* context = nullptr);

// Offset of row `row`'s scale: the C-order element `row` of the scales array, through its strides.
SD_HOST_DEVICE SD_INLINE LongType kvRowScaleOffset(const LongType* scalesShapeInfo, const LongType row) {
    LongType coords[SD_MAX_RANK];
    const LongType scalesRank = shape::rank(scalesShapeInfo);
    INDEX2COORDS(row, scalesRank, shape::shapeOf(scalesShapeInfo), coords);
    LongType offset;
    COORDS2INDEX(scalesRank, shape::stride(scalesShapeInfo), coords, offset);
    return offset;
}

// INT4 value of an already-scaled element: clamp to the symmetric range [-7, 7] and round half to
// even, as INT8 quantization does. Shared by the CPU and CUDA kernels so both backends pack the
// same nibbles.
template <typename AccT>
SD_HOST_DEVICE SD_INLINE int kvQuantizeInt4Value(AccT val) {
    val = sd::math::sd_max<AccT>(static_cast<AccT>(-7), sd::math::sd_min<AccT>(static_cast<AccT>(7), val));
    return static_cast<int>(sd::math::sd_rint<AccT, AccT>(val));
}

/**
 * V2 quantised-on-write helper: append one decode-step K or V vector into a fixed-allocation
 * INT8 cache buffer and write the per-token-per-head scale.
 *
 * Called once per attention layer per decode step (inside dot_product_attention_v2 on the
 * QUANTIZED path). CUDA-graph safe: all pointer addresses are stable; the scatter position is
 * read from device-side cachePosPtr at kernel runtime (same pattern as kvInPlaceWriteBSHD).
 *
 * Layout contract (INT8_KV mode):
 *   newKv      : T       [batch, 1,        kvHeads, headDim]  — current-step K or V, any float type
 *   quantCache : INT8    [batch, maxKvLen, kvHeads, headDim + 4] — row-inline, scale in the last 4 bytes
 *              | INT8    [batch, maxKvLen, kvHeads, headDim]     — with a separate scaleCache
 *   scaleCache : FLOAT32 [batch, maxKvLen, kvHeads] or nullptr for row-inline
 *   cachePosPtr: int64 write position (device-resident on CUDA, host on CPU); an out-of-range
 *                position writes nothing
 *
 * Per-row (per KV head) absmax in AccT + INT8 scatter + scale scatter. Every operand is
 * addressed through its own strides. No cross-row atomics → capture-safe.
 */
SD_LIB_HIDDEN void kvInPlaceWriteQuantisedBSHD(
    NDArray* quantCache,     // INT8 [batch, maxKvLen, kvHeads, headDim(+4)] — modified in-place
    NDArray* scaleCache,     // FLOAT32 [batch, maxKvLen, kvHeads] or nullptr — modified in-place
    NDArray* newKv,          // T [batch, 1, kvHeads, headDim]               — source (current step)
    const void* cachePosPtr, // pointer to int64 write position
    LaunchContext* context);

/**
 * CPU single-token GQA decode over INT8 K/V caches, with the same contract as the CUDA
 * fusedGQADecodeQuantisedCuda (see FlashAttentionHelper.h): the query, bias, current windows
 * and output share the model dtype T; each cache row is dequantized with its FLOAT32 scale;
 * scores and softmax accumulate in AggregateType<T>.
 *
 * quantKeyCache  : INT8    [batch, seqKV, kvHeads, headDim] or [.., headDim + 4] row-inline
 * keyScaleCache  : FLOAT32 [batch, seqKV, kvHeads] or nullptr (row-inline)
 * quantValCache  : INT8    [batch, seqKV, kvHeads, headDim] or [.., headDim + 4] row-inline
 * valScaleCache  : FLOAT32 [batch, seqKV, kvHeads] or nullptr (row-inline)
 * query          : T       [batch, 1,     qHeads,  headDim]
 * output         : T       [batch, 1,     qHeads,  headDim]
 * attentionBias  : T       [batch, 1/qHeads, 1, seqKV] (rank 1-3 broadcast) or nullptr
 */
// currentKeyWindow/currentValueWindow/currentKvPosition: the pre-quantization current-step K/V
// rows (T [batch, currentSeq, kvHeads, headDim]) plus their host cache position. When non-null,
// attention covers only the written prefix [0, currentKvPosition + currentSeq), independent of
// attentionBias, and the window rows are read directly instead of round-tripping through the INT8
// cache. nullptr attends over the full seqKV and relies on the bias alone to mask.
// attentionScores/attentionLogits: optional T [batch, qHeads, 1, seqKV] outputs; null or empty
// means not requested. Logits are the pre-softmax scores, scores the softmax weights applied to V;
// rows past the written prefix get the masked logit and a zero score.
SD_LIB_HIDDEN void fusedGQADecodeQuantisedCpu(
    NDArray* query,
    NDArray* quantKeyCache,
    NDArray* keyScaleCache,
    NDArray* quantValCache,
    NDArray* valScaleCache,
    NDArray* output,
    double scale,
    NDArray* attentionBias,
    LaunchContext* context,
    NDArray* currentKeyWindow = nullptr,
    NDArray* currentValueWindow = nullptr,
    const void* currentKvPosition = nullptr,
    NDArray* attentionScores = nullptr,
    NDArray* attentionLogits = nullptr);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
