/* ******************************************************************************
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
// Paged attention kernel for block-based KV cache.
//
// Instead of dense [batch, seqLen, heads, headDim] KV tensors, this kernel
// gathers K/V from scattered physical blocks via a page table indirection.
// This enables:
// - O(1) slot reassignment for continuous batching (swap page tables, not data)
// - No per-sequence maxSeqLen pre-allocation (blocks allocated on demand)
// - Immediate memory reuse when sequences finish (return blocks to pool)
//

#ifndef LIBND4J_PAGED_ATTENTION_H
#define LIBND4J_PAGED_ATTENTION_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Configuration for paged attention.
 */
struct PagedAttentionConfig {
    int blockSize;          // Tokens per block (default 64)
    int numHeads;           // Number of query heads
    int numKvHeads;         // Number of KV heads (for GQA; 0 = same as numHeads)
    int headDim;            // Dimension per head
    float scale;            // Attention scale (0 = auto: 1/sqrt(headDim))
    bool isCausal;          // Apply causal masking

    PagedAttentionConfig()
        : blockSize(64), numHeads(0), numKvHeads(0), headDim(0),
          scale(0.0f), isCausal(true) {}
};

/**
 * Paged attention forward pass.
 *
 * Computes scaled dot-product attention with K/V gathered from a block pool
 * via page table indirection.
 *
 * @param query         [batch, 1, numHeads, headDim] - current token queries
 * @param keyBlockPool  [numBlocks, blockSize, numKvHeads, headDim] - all key blocks
 * @param valueBlockPool [numBlocks, blockSize, numKvHeads, headDim] - all value blocks
 * @param pageTables    [batch, maxBlocksPerSeq] int32 - logical→physical block mapping
 * @param contextLens   [batch] int32 - number of valid tokens per sequence
 * @param output        [batch, 1, numHeads, headDim] - attention output
 * @param config        Attention configuration
 * @param context       Launch context (stream, etc.)
 */
void pagedAttentionForward(
    NDArray* query,
    NDArray* keyBlockPool,
    NDArray* valueBlockPool,
    NDArray* pageTables,
    NDArray* contextLens,
    NDArray* output,
    const PagedAttentionConfig& config,
    LaunchContext* context = nullptr);

/**
 * Append new KV entries to the block pool.
 *
 * Copies new key/value tokens into the correct physical blocks as specified
 * by the page table. Handles block boundaries (tokens that span two blocks).
 *
 * @param keyBlockPool  [numBlocks, blockSize, numKvHeads, headDim] - key blocks (modified)
 * @param valueBlockPool [numBlocks, blockSize, numKvHeads, headDim] - value blocks (modified)
 * @param newKeys       [batch, newLen, numKvHeads, headDim] - new key entries
 * @param newValues     [batch, newLen, numKvHeads, headDim] - new value entries
 * @param pageTables    [batch, maxBlocksPerSeq] int32 - page tables
 * @param contextLens   [batch] int32 - current valid lengths (before append)
 * @param blockSize     tokens per block
 * @param context       Launch context
 */
void pagedKvCacheAppend(
    NDArray* keyBlockPool,
    NDArray* valueBlockPool,
    NDArray* newKeys,
    NDArray* newValues,
    NDArray* pageTables,
    NDArray* contextLens,
    int blockSize,
    LaunchContext* context = nullptr);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_PAGED_ATTENTION_H
