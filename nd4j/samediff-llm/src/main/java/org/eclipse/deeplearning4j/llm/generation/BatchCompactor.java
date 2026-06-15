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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Compacts a batch by removing finished sequences to reduce computation.
 *
 * <p>During autoregressive generation, sequences that hit EOS early waste compute
 * in subsequent decode steps. BatchCompactor removes these sequences from the
 * batch tensors (embeddings, KV cache, attention masks) and tracks the original
 * indices for result reassembly.</p>
 *
 * <p>Compaction is only triggered when a configurable fraction of the batch has
 * finished (default: 25%), since the overhead of slicing tensors is only worthwhile
 * when enough sequences can be removed.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * BatchCompactor compactor = new BatchCompactor(originalBatchSize);
 *
 * // During decode loop, after checking EOS:
 * if (compactor.shouldCompact(finished)) {
 *     embeddings = compactor.compactBatch(embeddings, finished);
 *     kvCache = compactor.compactKvCache(kvCache, finished);
 *     // Update local batch tracking...
 * }
 *
 * // When assembling results, map back to original indices:
 * int originalIdx = compactor.getOriginalIndex(compactIdx);
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class BatchCompactor {

    private final int originalBatchSize;
    private int[] activeToOriginal;  // maps compact index -> original batch index
    private int currentBatchSize;
    private final double compactionThreshold;

    /**
     * Create a batch compactor.
     *
     * @param originalBatchSize the initial batch size
     * @param compactionThreshold fraction of finished sequences needed to trigger compaction (0.0-1.0)
     */
    public BatchCompactor(int originalBatchSize, double compactionThreshold) {
        this.originalBatchSize = originalBatchSize;
        this.currentBatchSize = originalBatchSize;
        this.compactionThreshold = compactionThreshold;

        this.activeToOriginal = new int[originalBatchSize];
        for (int i = 0; i < originalBatchSize; i++) {
            activeToOriginal[i] = i;
        }
    }

    /**
     * Create a batch compactor with default 25% compaction threshold.
     */
    public BatchCompactor(int originalBatchSize) {
        this(originalBatchSize, 0.25);
    }

    /**
     * Check if compaction should be triggered.
     *
     * @param finished array of finished flags (indexed by current compact indices)
     * @return true if enough sequences are finished to justify compaction
     */
    public boolean shouldCompact(boolean[] finished) {
        int finishedCount = 0;
        for (boolean f : finished) {
            if (f) finishedCount++;
        }
        // Don't compact if all finished or none finished
        if (finishedCount == 0 || finishedCount >= finished.length) return false;
        return (double) finishedCount / finished.length >= compactionThreshold;
    }

    /**
     * Compact a batch tensor by removing finished sequences.
     *
     * <p>The tensor's first dimension (batch) is sliced to keep only active sequences.
     * Works for tensors of any rank (embeddings [B,S,H], logits [B,V], etc.).</p>
     *
     * @param tensor the tensor to compact, shape [batchSize, ...]
     * @param finished which sequences are finished (current compact indices)
     * @return compacted tensor with finished sequences removed
     */
    public INDArray compactBatch(INDArray tensor, boolean[] finished) {
        if (tensor == null) return null;

        List<Integer> activeIndices = new ArrayList<>();
        for (int i = 0; i < finished.length; i++) {
            if (!finished[i]) {
                activeIndices.add(i);
            }
        }

        if (activeIndices.size() == finished.length) {
            return tensor; // Nothing to compact
        }

        // Build index array for selection
        INDArrayIndex[] indices = new INDArrayIndex[tensor.rank()];
        indices[0] = NDArrayIndex.indices(activeIndices.stream().mapToLong(Integer::longValue).toArray());
        for (int d = 1; d < tensor.rank(); d++) {
            indices[d] = NDArrayIndex.all();
        }

        return tensor.get(indices).dup();
    }

    /**
     * Compact the KV cache by removing finished sequences from all cache entries.
     *
     * <p>Each KV cache entry has shape [batch, numHeads, seqLen, headDim].
     * The batch dimension is sliced to keep only active sequences.</p>
     *
     * @param kvCache the current KV cache map (presentName -> tensor)
     * @param finished which sequences are finished
     * @return new KV cache map with compacted tensors
     */
    public Map<String, INDArray> compactKvCache(Map<String, INDArray> kvCache, boolean[] finished) {
        Map<String, INDArray> compacted = new HashMap<>();

        for (Map.Entry<String, INDArray> entry : kvCache.entrySet()) {
            INDArray compactedTensor = compactBatch(entry.getValue(), finished);
            compacted.put(entry.getKey(), compactedTensor);
        }

        return compacted;
    }

    /**
     * Perform compaction and update internal index mapping.
     *
     * <p>Call this once per compaction event. It updates the active-to-original
     * index mapping and returns the new batch size.</p>
     *
     * @param finished which sequences are finished (current compact indices)
     * @return the new (reduced) batch size
     */
    public int compact(boolean[] finished) {
        List<Integer> newActiveToOriginal = new ArrayList<>();
        for (int i = 0; i < finished.length; i++) {
            if (!finished[i]) {
                newActiveToOriginal.add(activeToOriginal[i]);
            }
        }

        int removedCount = currentBatchSize - newActiveToOriginal.size();
        currentBatchSize = newActiveToOriginal.size();
        activeToOriginal = newActiveToOriginal.stream().mapToInt(Integer::intValue).toArray();

        log.info("Batch compacted: {} -> {} active sequences ({} removed)",
                finished.length, currentBatchSize, removedCount);

        return currentBatchSize;
    }

    /**
     * Get the original batch index for a compact index.
     *
     * @param compactIndex index in the current compacted batch
     * @return the original batch index
     */
    public int getOriginalIndex(int compactIndex) {
        return activeToOriginal[compactIndex];
    }

    /**
     * Get the current (possibly compacted) batch size.
     */
    public int getCurrentBatchSize() {
        return currentBatchSize;
    }

    /**
     * Get the original batch size.
     */
    public int getOriginalBatchSize() {
        return originalBatchSize;
    }

    /**
     * Check if the batch has been compacted from its original size.
     */
    public boolean isCompacted() {
        return currentBatchSize < originalBatchSize;
    }

    /**
     * Create a new finished array remapped for compact indices.
     * After compaction, all entries are false (only active sequences remain).
     */
    public boolean[] newFinishedArray() {
        return new boolean[currentBatchSize];
    }
}
