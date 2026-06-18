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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Disk-based third tier for KV cache offloading (GPU -> Host -> SSD).
 *
 * <p>When host memory is also constrained, KV cache blocks can be further evicted
 * to disk storage. This class provides async read/write operations using NIO
 * file channels for efficient I/O, with an LRU eviction policy for managing
 * the on-disk block limit.</p>
 *
 * <h3>Three-tier memory hierarchy</h3>
 * <pre>
 * GPU (hot):   [block pool] -- active sequences, fast access
 *                   | evict/restore (async DMA)
 * Host (warm):  [pinned buffers] -- idle sequences, async restore
 *                   | evict/restore (async disk I/O)
 * Disk (cold):  [SSD files] -- deep storage, prefetch for anticipated needs
 * </pre>
 *
 * <h3>File format</h3>
 * <p>Each block is stored as {@code block_${blockId}.kv} in the storage directory.
 * The file contains raw bytes: key data followed by value data, each of size
 * {@code blockSize * numKvHeads * headDim * dataType.width()}.</p>
 *
 * <h3>Async I/O</h3>
 * <p>All disk operations are performed asynchronously using a fixed-size thread pool.
 * The {@link #prefetchBlocks(List)} method initiates reads for blocks that are
 * anticipated to be needed soon, allowing disk latency to be overlapped with
 * computation.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see KVCacheHostOffloader
 * @see TieredKVCacheManager
 */
@Slf4j
public class KVCacheDiskOffloader implements AutoCloseable {

    /** Default number of I/O threads for async disk operations. */
    public static final int DEFAULT_IO_THREADS = 4;

    @Getter
    private final Path storageDir;

    @Getter
    private final int maxDiskBlocks;

    @Getter
    private final int numKvHeads;

    @Getter
    private final int headDim;

    @Getter
    private final DataType dataType;

    @Getter
    private final int blockSize;

    /** Size in bytes of one KV block (key or value). */
    private final long blockByteSize;

    /** Async I/O executor. */
    private final ExecutorService ioExecutor;

    /**
     * LRU-ordered map of block IDs stored on disk.
     * Access-ordered LinkedHashMap: least-recently-used at head.
     * Value is the file path for the block.
     */
    private final LinkedHashMap<Integer, Path> diskBlocks;

    /** Prefetch cache: blocks loaded from disk but not yet consumed. */
    private final Map<Integer, CompletableFuture<KVCacheHostOffloader.KVBlockPair>> prefetchCache;

    // Statistics
    private long writeCount;
    private long readCount;
    private long evictionCount;
    private long writeTimeMs;
    private long readTimeMs;

    /**
     * Create a disk-based KV cache offloader.
     *
     * @param storageDir    directory for block files (created if not exists)
     * @param maxDiskBlocks maximum number of blocks to store on disk
     * @param blockSize     tokens per KV cache block
     * @param numKvHeads    number of KV attention heads
     * @param headDim       dimension per head
     * @param dataType      data type for KV tensors
     */
    public KVCacheDiskOffloader(Path storageDir, int maxDiskBlocks, int blockSize,
                                 int numKvHeads, int headDim, DataType dataType) {
        this(storageDir, maxDiskBlocks, blockSize, numKvHeads, headDim, dataType, DEFAULT_IO_THREADS);
    }

    /**
     * Create a disk-based KV cache offloader with configurable thread count.
     *
     * @param storageDir    directory for block files (created if not exists)
     * @param maxDiskBlocks maximum number of blocks to store on disk
     * @param blockSize     tokens per KV cache block
     * @param numKvHeads    number of KV attention heads
     * @param headDim       dimension per head
     * @param dataType      data type for KV tensors
     * @param ioThreads     number of I/O threads
     */
    public KVCacheDiskOffloader(Path storageDir, int maxDiskBlocks, int blockSize,
                                 int numKvHeads, int headDim, DataType dataType, int ioThreads) {
        this.storageDir = storageDir;
        this.maxDiskBlocks = maxDiskBlocks;
        this.blockSize = blockSize;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.dataType = dataType;
        this.blockByteSize = (long) blockSize * numKvHeads * headDim * dataType.width();

        this.ioExecutor = Executors.newFixedThreadPool(ioThreads, r -> {
            Thread t = new Thread(r, "kv-disk-io");
            t.setDaemon(true);
            return t;
        });

        // Access-ordered for LRU behavior
        this.diskBlocks = new LinkedHashMap<>(16, 0.75f, true);
        this.prefetchCache = new java.util.concurrent.ConcurrentHashMap<>();

        // Ensure storage directory exists
        try {
            Files.createDirectories(storageDir);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create KV cache disk storage directory: " + storageDir, e);
        }

        log.info("KVCacheDiskOffloader: storageDir={}, maxBlocks={}, blockBytes={}, ioThreads={}",
                storageDir, maxDiskBlocks, blockByteSize, ioThreads);
    }

    /**
     * Asynchronously evict a KV block pair from host memory to disk.
     *
     * <p>Writes the key and value data sequentially to a single file.
     * If the disk block limit is reached, the LRU block is deleted first.</p>
     *
     * @param blockId  the physical block ID being evicted
     * @param hostKey  host-side key tensor [blockSize, numKvHeads, headDim]
     * @param hostValue host-side value tensor [blockSize, numKvHeads, headDim]
     * @return future that completes when the write is finished
     */
    public CompletableFuture<Void> evictToDisk(int blockId, INDArray hostKey, INDArray hostValue) {
        // Serialize to byte arrays on the calling thread to avoid NDArray lifecycle issues
        byte[] keyBytes = toBytes(hostKey);
        byte[] valueBytes = toBytes(hostValue);

        return CompletableFuture.runAsync(() -> {
            long startMs = System.currentTimeMillis();
            try {
                // Evict LRU disk block if at capacity
                synchronized (diskBlocks) {
                    while (diskBlocks.size() >= maxDiskBlocks) {
                        evictLruDiskBlock();
                    }
                }

                Path blockFile = getBlockFile(blockId);
                try (FileChannel channel = FileChannel.open(blockFile,
                        StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
                    ByteBuffer keyBuf = ByteBuffer.wrap(keyBytes);
                    ByteBuffer valBuf = ByteBuffer.wrap(valueBytes);
                    while (keyBuf.hasRemaining()) {
                        channel.write(keyBuf);
                    }
                    while (valBuf.hasRemaining()) {
                        channel.write(valBuf);
                    }
                    channel.force(false);
                }

                synchronized (diskBlocks) {
                    diskBlocks.put(blockId, blockFile);
                }

                writeCount++;
                writeTimeMs += System.currentTimeMillis() - startMs;
                log.trace("Evicted block {} to disk: {}", blockId, blockFile);

            } catch (IOException e) {
                log.error("Failed to write block {} to disk", blockId, e);
                throw new RuntimeException("Disk eviction failed for block " + blockId, e);
            }
        }, ioExecutor);
    }

    /**
     * Asynchronously restore a KV block pair from disk.
     *
     * <p>If the block was previously prefetched, returns the cached future.
     * Otherwise initiates a new async read.</p>
     *
     * @param blockId the physical block ID to restore
     * @return future containing the key-value pair, or a failed future if not on disk
     */
    public CompletableFuture<KVCacheHostOffloader.KVBlockPair> restoreFromDisk(int blockId) {
        // Check prefetch cache first
        CompletableFuture<KVCacheHostOffloader.KVBlockPair> prefetched = prefetchCache.remove(blockId);
        if (prefetched != null) {
            return prefetched;
        }

        return doAsyncRead(blockId);
    }

    /**
     * Check if a block exists on disk.
     *
     * @param blockId the physical block ID
     * @return true if the block is stored on disk
     */
    public boolean hasBlock(int blockId) {
        synchronized (diskBlocks) {
            return diskBlocks.containsKey(blockId);
        }
    }

    /**
     * Delete a block's disk file and remove from tracking.
     *
     * @param blockId the physical block ID to release
     */
    public void releaseBlock(int blockId) {
        Path blockFile;
        synchronized (diskBlocks) {
            blockFile = diskBlocks.remove(blockId);
        }
        if (blockFile != null) {
            try {
                Files.deleteIfExists(blockFile);
                log.trace("Deleted disk block {}: {}", blockId, blockFile);
            } catch (IOException e) {
                log.warn("Failed to delete disk block file: {}", blockFile, e);
            }
        }
        // Also cancel any pending prefetch
        CompletableFuture<KVCacheHostOffloader.KVBlockPair> pending = prefetchCache.remove(blockId);
        if (pending != null) {
            pending.cancel(false);
        }
    }

    /**
     * Initiate async reads for blocks anticipated to be needed soon.
     *
     * <p>Prefetched blocks are cached in memory until consumed by
     * {@link #restoreFromDisk(int)}. Already-prefetched or missing blocks
     * are skipped.</p>
     *
     * @param blockIds block IDs to prefetch
     */
    public void prefetchBlocks(List<Integer> blockIds) {
        for (int blockId : blockIds) {
            if (prefetchCache.containsKey(blockId)) {
                continue; // Already prefetching
            }
            synchronized (diskBlocks) {
                if (!diskBlocks.containsKey(blockId)) {
                    continue; // Not on disk
                }
            }
            prefetchCache.put(blockId, doAsyncRead(blockId));
        }
    }

    /**
     * Get the number of blocks currently stored on disk.
     */
    public int getDiskBlockCount() {
        synchronized (diskBlocks) {
            return diskBlocks.size();
        }
    }

    /**
     * Get performance statistics.
     */
    public String getStats() {
        double avgWriteMs = writeCount > 0 ? (double) writeTimeMs / writeCount : 0;
        double avgReadMs = readCount > 0 ? (double) readTimeMs / readCount : 0;
        synchronized (diskBlocks) {
            return String.format(
                    "KVCacheDiskOffloader[onDisk=%d/%d, writes=%d(avg=%.1fms), reads=%d(avg=%.1fms), evictions=%d]",
                    diskBlocks.size(), maxDiskBlocks, writeCount, avgWriteMs,
                    readCount, avgReadMs, evictionCount);
        }
    }

    @Override
    public void close() {
        // Shutdown executor
        ioExecutor.shutdown();
        try {
            if (!ioExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                ioExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            ioExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // Cancel pending prefetches
        for (CompletableFuture<KVCacheHostOffloader.KVBlockPair> future : prefetchCache.values()) {
            future.cancel(false);
        }
        prefetchCache.clear();

        // Delete all block files
        synchronized (diskBlocks) {
            for (Path blockFile : diskBlocks.values()) {
                try {
                    Files.deleteIfExists(blockFile);
                } catch (IOException e) {
                    log.warn("Failed to delete disk block file during close: {}", blockFile, e);
                }
            }
            diskBlocks.clear();
        }

        // Attempt to delete the storage directory
        try {
            Files.deleteIfExists(storageDir);
        } catch (IOException e) {
            log.debug("Could not delete storage directory (may not be empty): {}", storageDir);
        }

        log.info("KVCacheDiskOffloader closed. Final stats: {}", getStats());
    }

    // ========== Internal Helpers ==========

    private Path getBlockFile(int blockId) {
        return storageDir.resolve("block_" + blockId + ".kv");
    }

    /**
     * Evict the least-recently-used block from disk.
     * Must be called while holding the diskBlocks lock.
     */
    private void evictLruDiskBlock() {
        if (diskBlocks.isEmpty()) return;

        Map.Entry<Integer, Path> lruEntry = diskBlocks.entrySet().iterator().next();
        int lruBlockId = lruEntry.getKey();
        Path lruFile = lruEntry.getValue();

        diskBlocks.remove(lruBlockId);
        evictionCount++;

        try {
            Files.deleteIfExists(lruFile);
            log.trace("Disk-evicted LRU block {}", lruBlockId);
        } catch (IOException e) {
            log.warn("Failed to delete evicted disk block: {}", lruFile, e);
        }
    }

    private CompletableFuture<KVCacheHostOffloader.KVBlockPair> doAsyncRead(int blockId) {
        return CompletableFuture.supplyAsync(() -> {
            long startMs = System.currentTimeMillis();
            Path blockFile;
            synchronized (diskBlocks) {
                blockFile = diskBlocks.get(blockId);
            }
            if (blockFile == null || !Files.exists(blockFile)) {
                throw new RuntimeException("Block " + blockId + " not found on disk");
            }

            try (FileChannel channel = FileChannel.open(blockFile, StandardOpenOption.READ)) {
                byte[] keyBytes = new byte[(int) blockByteSize];
                byte[] valueBytes = new byte[(int) blockByteSize];

                ByteBuffer keyBuf = ByteBuffer.wrap(keyBytes);
                while (keyBuf.hasRemaining()) {
                    if (channel.read(keyBuf) == -1) break;
                }
                ByteBuffer valBuf = ByteBuffer.wrap(valueBytes);
                while (valBuf.hasRemaining()) {
                    if (channel.read(valBuf) == -1) break;
                }

                INDArray key = fromBytes(keyBytes);
                INDArray value = fromBytes(valueBytes);

                readCount++;
                readTimeMs += System.currentTimeMillis() - startMs;
                log.trace("Restored block {} from disk", blockId);

                return new KVCacheHostOffloader.KVBlockPair(key, value, blockId);

            } catch (IOException e) {
                throw new RuntimeException("Failed to read block " + blockId + " from disk", e);
            }
        }, ioExecutor);
    }

    /**
     * Serialize an INDArray to raw bytes.
     */
    private byte[] toBytes(INDArray array) {
        INDArray flat = array.isView() ? array.dup() : array;
        java.nio.ByteBuffer buffer = flat.data().asNio();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return bytes;
    }

    /**
     * Deserialize raw bytes to an INDArray with the configured block shape.
     */
    private INDArray fromBytes(byte[] bytes) {
        INDArray array = Nd4j.create(dataType, blockSize, numKvHeads, headDim);
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        ByteBuffer dest = array.data().asNio();
        dest.put(buffer);
        // Tag as HOST so CUDA backend syncs host→device on next access
        Nd4j.getAffinityManager().tagLocation(array, AffinityManager.Location.HOST);
        return array;
    }

    private INDArray createBuffer(byte[] bytes) {
        return Nd4j.create(dataType, blockSize, numKvHeads, headDim);
    }
}
