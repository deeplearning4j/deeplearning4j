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

//
// @author Eclipse Deeplearning4j
//
// CPU implementation of moe_weighted_sum helpers.
//

#include <execution/Threads.h>
#include <ops/declarable/helpers/moe_weighted_sum.h>
#include <system/op_boilerplate.h>
#include <system/selective_rendering.h>
#include <type_traits>

#if NOT_EXCLUDED(OP_moe_weighted_sum)

namespace sd {
namespace ops {
namespace helpers {

// ─────────────────────────────────────────────────────────────────────────────
// Dense layout  [T, topK, D]
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
static void moeWeightedSumDense_(sd::LaunchContext* context,
                                  NDArray* expertOutputs,
                                  NDArray* weights,
                                  NDArray*       output) {
    // expertOutputs: [T, topK, D]
    // weights:       [T, topK]
    // output:        [T, D]

    const sd::LongType numTokens = expertOutputs->sizeAt(0);
    const sd::LongType topK      = expertOutputs->sizeAt(1);
    const sd::LongType D         = expertOutputs->sizeAt(2);

    // Strides
    const sd::LongType eoStrideT    = expertOutputs->strideAt(0);
    const sd::LongType eoStrideK    = expertOutputs->strideAt(1);
    const sd::LongType eoStrideD    = expertOutputs->strideAt(2);

    const sd::LongType wStrideT     = weights->strideAt(0);
    const sd::LongType wStrideK     = weights->strideAt(1);

    const sd::LongType outStrideT   = output->strideAt(0);
    const sd::LongType outStrideD   = output->strideAt(1);

    const T* eoPtr  = expertOutputs->bufferAsT<T>();
    const T* wPtr   = weights->bufferAsT<T>();
    T*       outPtr = output->bufferAsT<T>();

    // Parallel over T tokens; accumulate topK experts in fp32
    auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t inc) {
        for (sd::LongType t = start; t < stop; t += inc) {
            T* out_t = outPtr + t * outStrideT;

            // Zero the output row for this token
            for (sd::LongType d = 0; d < D; d++) {
                out_t[d * outStrideD] = static_cast<T>(0);
            }

            for (sd::LongType k = 0; k < topK; k++) {
                // fp32 weight accumulation
                const float w = static_cast<float>(wPtr[t * wStrideT + k * wStrideK]);
                const T* eo   = eoPtr + t * eoStrideT + k * eoStrideK;

                for (sd::LongType d = 0; d < D; d++) {
                    // fp32 accumulation even for fp16 inputs
                    const float val = static_cast<float>(eo[d * eoStrideD]);
                    out_t[d * outStrideD] = static_cast<T>(
                        static_cast<float>(out_t[d * outStrideD]) + w * val);
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, numTokens, 1);
}

void moeWeightedSumDense(sd::LaunchContext* context,
                          NDArray* expertOutputs,
                          NDArray* weights,
                          NDArray*       output) {
    BUILD_SINGLE_SELECTOR(expertOutputs->dataType(), moeWeightedSumDense_,
                          (context, expertOutputs, weights, output),
                          SD_FLOAT_TYPES);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sorted-flat layout  [totalRows, D]
// ─────────────────────────────────────────────────────────────────────────────

// INT index type dispatch helper — the row-index can be INT32 or INT64.
template <typename T, typename IdxT>
static void moeWeightedSumFlat_(sd::LaunchContext* context,
                                 NDArray* expertOutputs,
                                 NDArray* weights,
                                 NDArray* rowIndex,
                                 NDArray*       output) {
    // expertOutputs: [totalRows, D]
    // weights:       [T, topK]
    // rowIndex:      [totalRows, 2]  — {token_idx, k_idx} per row
    // output:        [T, D]

    const sd::LongType totalRows = expertOutputs->sizeAt(0);
    const sd::LongType D         = expertOutputs->sizeAt(1);

    const sd::LongType eoStrideRow = expertOutputs->strideAt(0);
    const sd::LongType eoStrideD   = expertOutputs->strideAt(1);

    const sd::LongType wStrideT    = weights->strideAt(0);
    const sd::LongType wStrideK    = weights->strideAt(1);

    const sd::LongType riStrideRow = rowIndex->strideAt(0);  // stride between rows
    // rowIndex->strideAt(1) is 1 for contiguous layout

    const sd::LongType outStrideT  = output->strideAt(0);
    const sd::LongType outStrideD  = output->strideAt(1);

    const T*    eoPtr  = expertOutputs->bufferAsT<T>();
    const T*    wPtr   = weights->bufferAsT<T>();
    const IdxT* riPtr  = rowIndex->bufferAsT<IdxT>();
    T*          outPtr = output->bufferAsT<T>();

    // Zero output first (parallel, cache-friendly)
    output->nullify();

    // Sequential scatter-add over rows.  For large totalRows the bottleneck is
    // the atomic add into the output; for the typical MoE sizes (T=1024,
    // topK=8, D=4096) the working set fits in L3 and the sequential scan is
    // competitive with a sorted approach.  An OpenMP reduction or a two-pass
    // approach can be introduced if profiling shows this as a hot path.
    for (sd::LongType r = 0; r < totalRows; r++) {
        const sd::LongType token_idx = static_cast<sd::LongType>(riPtr[r * riStrideRow + 0]);
        const sd::LongType k_idx     = static_cast<sd::LongType>(riPtr[r * riStrideRow + 1]);

        const float w    = static_cast<float>(wPtr[token_idx * wStrideT + k_idx * wStrideK]);
        const T*    eo_r = eoPtr + r * eoStrideRow;
        T*          out  = outPtr + token_idx * outStrideT;

        for (sd::LongType d = 0; d < D; d++) {
            const float val = static_cast<float>(eo_r[d * eoStrideD]);
            out[d * outStrideD] = static_cast<T>(
                static_cast<float>(out[d * outStrideD]) + w * val);
        }
    }
}

// Outer dispatch: choose index type then element type
template <typename T>
static void moeWeightedSumFlatDispatchIdx_(sd::LaunchContext* context,
                                            NDArray* expertOutputs,
                                            NDArray* weights,
                                            NDArray* rowIndex,
                                            NDArray*       output) {
    if (rowIndex->dataType() == sd::DataType::INT64) {
        moeWeightedSumFlat_<T, sd::LongType>(context, expertOutputs, weights, rowIndex, output);
    } else {
        // INT32
        moeWeightedSumFlat_<T, int>(context, expertOutputs, weights, rowIndex, output);
    }
}

void moeWeightedSumFlat(sd::LaunchContext* context,
                         NDArray* expertOutputs,
                         NDArray* weights,
                         NDArray* rowIndex,
                         NDArray*       output) {
    BUILD_SINGLE_SELECTOR(expertOutputs->dataType(), moeWeightedSumFlatDispatchIdx_,
                          (context, expertOutputs, weights, rowIndex, output),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_moe_weighted_sum)
