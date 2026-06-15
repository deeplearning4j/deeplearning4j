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

package org.eclipse.deeplearning4j.llm.generation.kvcache;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Three-tier KV cache manager composing GPU, Host, and Disk tiers.
 *
 * <p>This manager coordinates memory pressure across all three tiers,
 * cascading evictions from hot (GPU) to warm (host) to cold (disk)
 * storage. Restoration follows the reverse path, pulling from the
 * deepest available tier.</p>
 *
 * <h3>Pressure thresholds</h3>
 * <ul>
 *   <li>GPU: evict when free blocks &lt; 10% of total pool</li>
 *   <li>Host: evict to disk when host pool is full</li>
 * </ul>
 *
 * <h3>Priority ordering</h3>
 * <ul>
 *   <li>GPU hot: actively used by running sequences</li>
 *   <li>Host warm: recently evicted, fast to restore</li>
 *   <li>Disk cold: deeply evicted, requires async I/O</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see PagedKVCache
 * @see KVCacheHostOffloader
 * @see KVCacheDiskOffloader
 */
@Slf4j
public class TieredKVCacheManager {

    /** Default GPU pressure threshold: evict when less than 10% free. */
    public static final double DEFAULT_GPU_PRESSURE_THRESHOLD = 0.10;

    @Getter
    private final PagedKVCache gpuCache;

    @Getter
    private final KVCacheHostOffloader hostOffloader;

    @Getter
    private final KVCacheDiskOffloader diskOffloader;

    /** Fraction of free GPU blocks below which pressure eviction triggers. */
    @Getter
    private final double gpuPressureThreshold;

    /**
     * LRU tracking for GPU blocks: maps blockId to sequence index.
     * Least-recently-used at head (access-ordered LinkedHashMap).
     */
    private final LinkedHashMap<Integer, Integer> gpuBlockLru;

    /**
     * LRU tracking for host blocks: maps blockId to timestamp.
     * Least-recently-used at head.
     */
    private final LinkedHashMap<Integer, Long> hostBlockLru;

    /**
     * Create a tiered KV cache manager with default pressure threshold.
     *
     * @param gpuCache      the GPU-side paged KV cache
     * @param hostOffloader the host-side offloader
     * @param diskOffloader the disk-side offloader
     */
    public TieredKVCacheManager(PagedKVCache gpuCache, KVCacheHostOffloader hostOffloader,
                                 KVCacheDiskOffloader diskOffloader) {
        this(gpuCache, hostOffloader, diskOffloader, DEFAULT_GPU_PRESSURE_THRESHOLD);
    }

    /**
     * Create a tiered KV cache manager with custom pressure threshold.
     *
     * @param gpuCache              the GPU-side paged KV cache
     * @param hostOffloader         the host-side offloader
     * @param diskOffloader         the disk-side offloader
     * @param gpuPressureThreshold  fraction of free blocks below which to trigger eviction
     */
    public TieredKVCacheManager(PagedKVCache gpuCache, KVCacheHostOffloader hostOffloader,
                                 KVCacheDiskOffloader diskOffloader, double gpuPressureThreshold) {
        this.gpuCache = gpuCache;
        this.hostOffloader = hostOffloader;
        this.diskOffloader = diskOffloader;
        this.gpuPressureThreshold = gpuPressureThreshold;

        this.gpuBlockLru = new LinkedHashMap<>(16, 0.75f, true);
        this.hostBlockLru = new LinkedHashMap<>(16, 0.75f, true);
    }

    /**
     * Record a GPU block access for LRU tracking.
     *
     * @param blockId the accessed block ID
     * @param seqIdx  the sequence that accessed it
     */
    public void touchGpuBlock(int blockId, int seqIdx) {
        gpuBlockLru.put(blockId, seqIdx);
    }

    /**
     * Record a host block access for LRU tracking.
     *
     * @param blockId the accessed block ID
     */
    public void touchHostBlock(int blockId) {
        hostBlockLru.put(blockId, System.nanoTime());
    }

    /**
     * Check memory pressure across tiers and cascade evictions as needed.
     *
     * <p>If the GPU pool has less than {@code gpuPressureThreshold} fraction
     * of blocks free, evicts the LRU GPU blocks to host. If host is also
     * full, cascades further to disk.</p>
     */
    public void checkPressure() {
        int totalBlocks = gpuCache.getNumBlocks();
        int freeBlocks = gpuCache.getNumFreeBlocks();
        double freeRatio = (double) freeBlocks / totalBlocks;

        if (freeRatio < gpuPressureThreshold) {
            // Calculate how many blocks to evict to reach threshold + 5% headroom
            int targetFree = (int) Math.ceil(totalBlocks * (gpuPressureThreshold + 0.05));
            int toEvict = targetFree - freeBlocks;

            log.info("GPU pressure detected: {}/{} free ({}%), evicting {} blocks",
                    freeBlocks, totalBlocks, String.format("%.1f", freeRatio * 100), toEvict);

            List<Integer> gpuEvictionCandidates = getGpuLruBlockIds(toEvict);
            for (int blockId : gpuEvictionCandidates) {
                evictFromGpu(blockId);
            }
        }

        // Check host pressure: if host is at capacity and disk is available
        if (hostOffloader.getHostBlockCount() >= hostOffloader.getMaxHostBlocks()) {
            // Evict oldest host blocks to disk
            List<Integer> hostEvictionCandidates = getHostLruBlockIds(
                    Math.max(1, hostOffloader.getMaxHostBlocks() / 20)); // Evict 5%
            for (int blockId : hostEvictionCandidates) {
                evictFromHost(blockId);
            }
        }
    }

    /**
     * Evict a specific GPU block to host memory.
     *
     * <p>Copies the block's key and value data from the GPU block pool to
     * a host-side buffer via the host offloader, then frees the GPU block.</p>
     *
     * @param blockId the physical block ID to evict from GPU
     */
    public void evictFromGpu(int blockId) {
        // Extract key/value from GPU pool for this block
        INDArray gpuKey = gpuCache.getKeyBlockPool().get(
                NDArrayIndex.point(blockId), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray gpuValue = gpuCache.getValueBlockPool().get(
                NDArrayIndex.point(blockId), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());

        // Evict to host (copies data)
        hostOffloader.evictBlock(blockId, gpuKey, gpuValue);
        touchHostBlock(blockId);

        // Remove from GPU LRU tracking
        gpuBlockLru.remove(blockId);

        log.trace("Evicted block {} from GPU to host", blockId);
    }

    /**
     * Evict a specific host block to disk storage.
     *
     * @param blockId the physical block ID to evict from host
     */
    public void evictFromHost(int blockId) {
        KVCacheHostOffloader.KVBlockPair pair = hostOffloader.restoreBlock(blockId);
        if (pair == null) {
            log.trace("Block {} not in host memory, cannot evict to disk", blockId);
            return;
        }

        // Async write to disk
        diskOffloader.evictToDisk(blockId, pair.getKey(), pair.getValue());

        // Release from host
        hostOffloader.releaseBlock(blockId);
        hostBlockLru.remove(blockId);

        log.trace("Evicted block {} from host to disk", blockId);
    }

    /**
     * Restore a block from the deepest available tier back to GPU.
     *
     * <p>Searches tiers in order: host first (fast), then disk (slow).
     * Returns a future that completes when the block data is available
     * in host memory, ready to be copied to the GPU block pool.</p>
     *
     * @param blockId the physical block ID to restore
     * @return future containing the restored KV block pair, or a failed future if not found
     */
    public CompletableFuture<Void> restoreBlock(int blockId) {
        // Try host first
        if (hostOffloader.hasBlock(blockId)) {
            KVCacheHostOffloader.KVBlockPair pair = hostOffloader.restoreBlock(blockId);
            if (pair != null) {
                copyToGpuPool(blockId, pair.getKey(), pair.getValue());
                hostOffloader.releaseBlock(blockId);
                hostBlockLru.remove(blockId);
                log.trace("Restored block {} from host to GPU", blockId);
                return CompletableFuture.completedFuture(null);
            }
        }

        // Try disk
        if (diskOffloader.hasBlock(blockId)) {
            return diskOffloader.restoreFromDisk(blockId).thenAccept(pair -> {
                copyToGpuPool(blockId, pair.getKey(), pair.getValue());
                pair.close();
                diskOffloader.releaseBlock(blockId);
                log.trace("Restored block {} from disk to GPU", blockId);
            });
        }

        return CompletableFuture.failedFuture(
                new IllegalStateException("Block " + blockId + " not found in any tier"));
    }

    /**
     * Get a summary of the current state across all tiers.
     */
    public String getStats() {
        int gpuUsed = gpuCache.getNumBlocks() - gpuCache.getNumFreeBlocks();
        return String.format(
                "TieredKVCache[GPU=%d/%d, Host=%d/%d, Disk=%d/%d]",
                gpuUsed, gpuCache.getNumBlocks(),
                hostOffloader.getHostBlockCount(), hostOffloader.getMaxHostBlocks(),
                diskOffloader.getDiskBlockCount(), diskOffloader.getMaxDiskBlocks());
    }

    // ========== Internal Helpers ==========

    /**
     * Get the N least-recently-used GPU block IDs.
     */
    private List<Integer> getGpuLruBlockIds(int count) {
        List<Integer> result = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : gpuBlockLru.entrySet()) {
            if (result.size() >= count) break;
            result.add(entry.getKey());
        }
        return result;
    }

    /**
     * Get the N least-recently-used host block IDs.
     */
    private List<Integer> getHostLruBlockIds(int count) {
        List<Integer> result = new ArrayList<>();
        for (Map.Entry<Integer, Long> entry : hostBlockLru.entrySet()) {
            if (result.size() >= count) break;
            result.add(entry.getKey());
        }
        return result;
    }

    /**
     * Copy key/value data into the GPU block pool at the given block position.
     */
    private void copyToGpuPool(int blockId, INDArray key, INDArray value) {
        INDArray destKey = gpuCache.getKeyBlockPool().get(
                NDArrayIndex.point(blockId), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        INDArray destValue = gpuCache.getValueBlockPool().get(
                NDArrayIndex.point(blockId), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());

        destKey.assign(key);
        destValue.assign(value);
        touchGpuBlock(blockId, -1);
    }
}
