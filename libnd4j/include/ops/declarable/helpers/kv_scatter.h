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

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
