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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Manager for multiple KV cache checkpoints with automatic lifecycle management.
 *
 * <p>Maintains an ordered collection of named checkpoints with a configurable
 * maximum count. When the limit is exceeded, the oldest checkpoints are
 * automatically deleted (FIFO eviction).</p>
 *
 * <h3>Checkpoint IDs</h3>
 * <p>Each checkpoint is assigned a unique UUID-based ID upon creation. This ID
 * is used for retrieval and deletion. IDs are returned by
 * {@link #createCheckpoint(Map, int)} and can be listed with
 * {@link #listCheckpoints()}.</p>
 *
 * <h3>Memory management</h3>
 * <p>Deleting a checkpoint via {@link #deleteCheckpoint(String)} frees all
 * INDArray buffers associated with that checkpoint. Checkpoints that are
 * evicted due to the max count limit are also properly closed.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see KVCacheCheckpoint
 */
@Slf4j
public class KVCacheCheckpointManager implements AutoCloseable {

    /** Default maximum number of checkpoints to retain. */
    public static final int DEFAULT_MAX_CHECKPOINTS = 16;

    @Getter
    private final int maxCheckpoints;

    /**
     * Insertion-ordered map of checkpoint ID to checkpoint.
     * Oldest entries are at the head.
     */
    private final LinkedHashMap<String, KVCacheCheckpoint> checkpoints;

    /** Ordered list of checkpoint IDs (by creation time). */
    private final List<String> checkpointOrder;

    /**
     * Create a checkpoint manager.
     *
     * @param maxCheckpoints maximum number of checkpoints to retain
     */
    public KVCacheCheckpointManager(int maxCheckpoints) {
        if (maxCheckpoints < 1) {
            throw new IllegalArgumentException("maxCheckpoints must be >= 1, got " + maxCheckpoints);
        }
        this.maxCheckpoints = maxCheckpoints;
        this.checkpoints = new LinkedHashMap<>();
        this.checkpointOrder = new ArrayList<>();
    }

    /**
     * Create a checkpoint manager with default maximum.
     */
    public KVCacheCheckpointManager() {
        this(DEFAULT_MAX_CHECKPOINTS);
    }

    /**
     * Create a new checkpoint from static KV cache buffers.
     *
     * <p>If the maximum number of checkpoints is exceeded, the oldest checkpoint
     * is automatically deleted.</p>
     *
     * @param kvBuffers     the KV cache buffers to snapshot
     * @param cachePosition current cache position
     * @return the checkpoint ID
     */
    public String createCheckpoint(Map<String, INDArray> kvBuffers, int cachePosition) {
        // Evict oldest if at capacity
        while (checkpoints.size() >= maxCheckpoints) {
            evictOldest();
        }

        String id = UUID.randomUUID().toString();
        KVCacheCheckpoint checkpoint = KVCacheCheckpoint.fromStaticKv(kvBuffers, cachePosition);

        checkpoints.put(id, checkpoint);
        checkpointOrder.add(id);

        log.debug("Created checkpoint {}: pos={}, memory={} bytes",
                id, cachePosition, checkpoint.getMemoryBytes());

        return id;
    }

    /**
     * Create a new checkpoint from a paged KV cache.
     *
     * @param cache  the paged KV cache
     * @param seqIdx the sequence index to snapshot
     * @return the checkpoint ID
     */
    public String createCheckpoint(PagedKVCache cache, int seqIdx) {
        while (checkpoints.size() >= maxCheckpoints) {
            evictOldest();
        }

        String id = UUID.randomUUID().toString();
        KVCacheCheckpoint checkpoint = KVCacheCheckpoint.fromPagedKv(cache, seqIdx);

        checkpoints.put(id, checkpoint);
        checkpointOrder.add(id);

        log.debug("Created paged checkpoint {}: seqLen={}", id, cache.getSequenceLength(seqIdx));

        return id;
    }

    /**
     * Retrieve a checkpoint by ID.
     *
     * @param id the checkpoint ID
     * @return the checkpoint, or null if not found
     */
    public KVCacheCheckpoint getCheckpoint(String id) {
        return checkpoints.get(id);
    }

    /**
     * Delete a checkpoint by ID and free its resources.
     *
     * @param id the checkpoint ID to delete
     */
    public void deleteCheckpoint(String id) {
        KVCacheCheckpoint removed = checkpoints.remove(id);
        checkpointOrder.remove(id);

        if (removed != null) {
            removed.close();
            log.debug("Deleted checkpoint {}", id);
        }
    }

    /**
     * List all checkpoint IDs ordered by creation time (oldest first).
     *
     * @return unmodifiable list of checkpoint IDs
     */
    public List<String> listCheckpoints() {
        return Collections.unmodifiableList(new ArrayList<>(checkpointOrder));
    }

    /**
     * Get the number of active checkpoints.
     */
    public int size() {
        return checkpoints.size();
    }

    /**
     * Get the total memory footprint of all checkpoints.
     */
    public long getTotalMemoryBytes() {
        long total = 0;
        for (KVCacheCheckpoint cp : checkpoints.values()) {
            total += cp.getMemoryBytes();
        }
        return total;
    }

    @Override
    public void close() {
        for (KVCacheCheckpoint checkpoint : checkpoints.values()) {
            checkpoint.close();
        }
        checkpoints.clear();
        checkpointOrder.clear();
        log.info("KVCacheCheckpointManager closed");
    }

    @Override
    public String toString() {
        return String.format("KVCacheCheckpointManager[checkpoints=%d/%d, memory=%d bytes]",
                checkpoints.size(), maxCheckpoints, getTotalMemoryBytes());
    }

    // ========== Internal Helpers ==========

    private void evictOldest() {
        if (checkpointOrder.isEmpty()) return;

        String oldestId = checkpointOrder.remove(0);
        KVCacheCheckpoint oldest = checkpoints.remove(oldestId);
        if (oldest != null) {
            oldest.close();
            log.debug("Evicted oldest checkpoint {}", oldestId);
        }
    }
}
