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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Immutable snapshot of KV cache state for checkpointing and rollback.
 *
 * <p>Captures the valid portion of KV cache buffers at a specific cache position.
 * Supports both static KV cache (dense pre-allocated buffers) and paged KV cache
 * (block-based) snapshots.</p>
 *
 * <h3>Static KV cache checkpointing</h3>
 * <p>For static KV caches, only the valid portion (up to cachePosition) of each
 * buffer is copied. This enables efficient fork/rollback for speculative decoding
 * and branch-and-bound search.</p>
 *
 * <h3>Incremental deltas</h3>
 * <p>The {@link #incrementalDelta(KVCacheCheckpoint)} method creates a checkpoint
 * that stores only the data added since a base checkpoint. This reduces memory
 * usage for frequent checkpoints during long generations.</p>
 *
 * <h3>Disk persistence</h3>
 * <p>Checkpoints can be saved to and loaded from disk for persistence across
 * restarts or for offline analysis.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see KVCacheCheckpointManager
 */
@Slf4j
public class KVCacheCheckpoint {

    /** The stored KV buffer snapshots, keyed by buffer name. */
    @Getter
    private final Map<String, INDArray> kvBuffers;

    /** Cache position at the time of snapshot. */
    @Getter
    private final int cachePosition;

    /** Base position for incremental deltas (0 for full checkpoints). */
    @Getter
    private final int basePosition;

    /** Whether this is an incremental delta (only stores data after basePosition). */
    @Getter
    private final boolean incremental;

    /** Paged cache page table snapshot (null for static KV checkpoints). */
    @Getter
    private final int[] pageTableSnapshot;

    /** Creation timestamp in milliseconds. */
    @Getter
    private final long createdAtMs;

    private KVCacheCheckpoint(Map<String, INDArray> kvBuffers, int cachePosition,
                               int basePosition, boolean incremental, int[] pageTableSnapshot) {
        this.kvBuffers = Collections.unmodifiableMap(kvBuffers);
        this.cachePosition = cachePosition;
        this.basePosition = basePosition;
        this.incremental = incremental;
        this.pageTableSnapshot = pageTableSnapshot;
        this.createdAtMs = System.currentTimeMillis();
    }

    /**
     * Create a checkpoint from static KV cache buffers.
     *
     * <p>Copies the valid portion (up to cachePosition) of each buffer.
     * KV buffers are expected to have shape [batch, numHeads, seqLen, headDim]
     * where seqLen is the pre-allocated maximum.</p>
     *
     * @param kvBuffers      the KV cache buffers (e.g., "present.0.key", "present.0.value")
     * @param cachePosition  number of valid token positions in the cache
     * @return a new checkpoint containing copies of the valid data
     */
    public static KVCacheCheckpoint fromStaticKv(Map<String, INDArray> kvBuffers, int cachePosition) {
        Map<String, INDArray> copies = new HashMap<>();

        for (Map.Entry<String, INDArray> entry : kvBuffers.entrySet()) {
            INDArray buffer = entry.getValue();
            if (buffer == null) continue;

            if (cachePosition > 0 && buffer.rank() == 4 && buffer.size(2) >= cachePosition) {
                // Copy only the valid portion: [batch, heads, 0:cachePosition, headDim]
                INDArray validSlice = buffer.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all());
                copies.put(entry.getKey(), validSlice.dup());
            } else if (cachePosition > 0) {
                // Non-standard shape, copy entirely
                copies.put(entry.getKey(), buffer.dup());
            }
            // If cachePosition == 0, skip (nothing to checkpoint)
        }

        return new KVCacheCheckpoint(copies, cachePosition, 0, false, null);
    }

    /**
     * Create a checkpoint from a paged KV cache for a specific sequence.
     *
     * <p>Snapshots the page table and copies the data from allocated blocks.</p>
     *
     * @param cache  the paged KV cache
     * @param seqIdx the sequence index to snapshot
     * @return a new checkpoint
     */
    public static KVCacheCheckpoint fromPagedKv(PagedKVCache cache, int seqIdx) {
        int seqLen = cache.getSequenceLength(seqIdx);
        int numAllocated = cache.getNumAllocatedBlocks(seqIdx);

        // Snapshot page table
        INDArray pageTable = cache.getPageTableArray(seqIdx);
        int[] pageTableArr = pageTable.toIntVector();
        pageTable.close();

        // Copy block data
        Map<String, INDArray> blockData = new HashMap<>();
        for (int b = 0; b < numAllocated; b++) {
            int blockId = pageTableArr[b];
            INDArray keyBlock = cache.getKeyBlockPool().get(
                    NDArrayIndex.point(blockId), NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.all());
            INDArray valueBlock = cache.getValueBlockPool().get(
                    NDArrayIndex.point(blockId), NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.all());

            blockData.put("key_block_" + b, keyBlock.dup());
            blockData.put("value_block_" + b, valueBlock.dup());
        }

        return new KVCacheCheckpoint(blockData, seqLen, 0, false, pageTableArr);
    }

    /**
     * Create an incremental delta checkpoint relative to a base checkpoint.
     *
     * <p>Only stores data from the base checkpoint's position to this checkpoint's
     * position. The base checkpoint must be from the same generation run.</p>
     *
     * @param base the base checkpoint to diff against
     * @return a new incremental checkpoint containing only the delta
     */
    public KVCacheCheckpoint incrementalDelta(KVCacheCheckpoint base) {
        if (base.cachePosition > this.cachePosition) {
            throw new IllegalArgumentException(
                    "Base checkpoint position (" + base.cachePosition +
                            ") is ahead of this checkpoint (" + this.cachePosition + ")");
        }

        Map<String, INDArray> deltaBuffers = new HashMap<>();

        for (Map.Entry<String, INDArray> entry : this.kvBuffers.entrySet()) {
            String name = entry.getKey();
            INDArray fullBuffer = entry.getValue();

            if (fullBuffer.rank() == 4) {
                long validLen = fullBuffer.size(2);
                if (base.cachePosition < validLen) {
                    // Extract only the delta: [batch, heads, basePos:currentPos, headDim]
                    INDArray delta = fullBuffer.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(base.cachePosition, validLen),
                            NDArrayIndex.all());
                    deltaBuffers.put(name, delta.dup());
                }
            } else {
                // For non-standard shapes, store the full buffer
                deltaBuffers.put(name, fullBuffer.dup());
            }
        }

        return new KVCacheCheckpoint(deltaBuffers, this.cachePosition,
                base.cachePosition, true, this.pageTableSnapshot);
    }

    /**
     * Save this checkpoint to disk in binary format.
     *
     * @param file the file to write to
     * @throws IOException if writing fails
     */
    public void saveToDisk(Path file) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(file))) {
            // Header
            dos.writeInt(cachePosition);
            dos.writeInt(basePosition);
            dos.writeBoolean(incremental);
            dos.writeLong(createdAtMs);

            // Page table (if present)
            if (pageTableSnapshot != null) {
                dos.writeInt(pageTableSnapshot.length);
                for (int id : pageTableSnapshot) {
                    dos.writeInt(id);
                }
            } else {
                dos.writeInt(-1);
            }

            // KV buffers
            dos.writeInt(kvBuffers.size());
            for (Map.Entry<String, INDArray> entry : kvBuffers.entrySet()) {
                dos.writeUTF(entry.getKey());
                INDArray arr = entry.getValue();

                // Write shape
                long[] shape = arr.shape();
                dos.writeInt(shape.length);
                for (long dim : shape) {
                    dos.writeLong(dim);
                }

                // Write data type
                dos.writeUTF(arr.dataType().name());

                // Write data
                java.nio.ByteBuffer buffer = arr.data().asNio();
                byte[] bytes = new byte[buffer.remaining()];
                buffer.get(bytes);
                dos.writeInt(bytes.length);
                dos.write(bytes);
            }
        }

        log.debug("Saved KVCacheCheckpoint to {}: pos={}, buffers={}", file, cachePosition, kvBuffers.size());
    }

    /**
     * Load a checkpoint from disk.
     *
     * @param file the file to read from
     * @return the loaded checkpoint
     * @throws IOException if reading fails
     */
    public static KVCacheCheckpoint loadFromDisk(Path file) throws IOException {
        try (DataInputStream dis = new DataInputStream(Files.newInputStream(file))) {
            int cachePosition = dis.readInt();
            int basePosition = dis.readInt();
            boolean incremental = dis.readBoolean();
            long createdAtMs = dis.readLong();

            // Page table
            int pageTableLen = dis.readInt();
            int[] pageTableSnapshot = null;
            if (pageTableLen >= 0) {
                pageTableSnapshot = new int[pageTableLen];
                for (int i = 0; i < pageTableLen; i++) {
                    pageTableSnapshot[i] = dis.readInt();
                }
            }

            // KV buffers
            int numBuffers = dis.readInt();
            Map<String, INDArray> kvBuffers = new HashMap<>();
            for (int b = 0; b < numBuffers; b++) {
                String name = dis.readUTF();

                // Read shape
                int rank = dis.readInt();
                long[] shape = new long[rank];
                for (int d = 0; d < rank; d++) {
                    shape[d] = dis.readLong();
                }

                // Read data type
                DataType dtype = DataType.valueOf(dis.readUTF());

                // Read data
                int byteLen = dis.readInt();
                byte[] bytes = new byte[byteLen];
                dis.readFully(bytes);

                INDArray arr = Nd4j.create(dtype, shape);
                java.nio.ByteBuffer buf = arr.data().asNio();
                buf.put(bytes);
                // Tag as HOST so CUDA backend syncs host→device on next access
                Nd4j.getAffinityManager().tagLocation(arr, AffinityManager.Location.HOST);

                kvBuffers.put(name, arr);
            }

            log.debug("Loaded KVCacheCheckpoint from {}: pos={}, buffers={}", file, cachePosition, numBuffers);

            return new KVCacheCheckpoint(kvBuffers, cachePosition, basePosition,
                    incremental, pageTableSnapshot);
        }
    }

    /**
     * Get the approximate memory footprint of this checkpoint in bytes.
     */
    public long getMemoryBytes() {
        long total = 0;
        for (INDArray arr : kvBuffers.values()) {
            total += arr.length() * arr.dataType().width();
        }
        return total;
    }

    /**
     * Close and free all stored arrays.
     */
    public void close() {
        for (INDArray arr : kvBuffers.values()) {
            if (arr != null && !arr.wasClosed()) {
                arr.close();
            }
        }
    }

    @Override
    public String toString() {
        return String.format("KVCacheCheckpoint[pos=%d, base=%d, incremental=%s, buffers=%d, bytes=%d]",
                cachePosition, basePosition, incremental, kvBuffers.size(), getMemoryBytes());
    }
}
