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

package org.nd4j.autodiff.samediff.serde;

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.HybridDataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.TransferMetrics;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;

/**
 * Scoped context for intelligent model loading with background transfer monitoring.
 *
 * <p>This class wraps model loading operations to provide:
 * <ul>
 *   <li>Pre-analysis of model size from manifest</li>
 *   <li>Intelligent target device selection using existing DeviceMemoryManager</li>
 *   <li>Background async transfers using HybridDataBuffer.prefetch()</li>
 *   <li>Integration with existing TransferMetrics for monitoring</li>
 *   <li>Progress callbacks for loading status</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * // Simple usage - automatic device selection
 * try (ModelLoadingContext ctx = ModelLoadingContext.forModel(new File("model.sdz"))) {
 *     SameDiff model = SDZSerializer.load(modelFile, false, ctx);
 * }
 * // ctx.close() logs TransferMetrics summary
 *
 * // With explicit device preference
 * try (ModelLoadingContext ctx = ModelLoadingContext.builder()
 *         .targetDevice(DeviceDescriptor.cuda(0))
 *         .build()) {
 *     SameDiff model = SDZSerializer.load(modelFile, false, ctx);
 * }
 *
 * // Monitor loading progress
 * try (ModelLoadingContext ctx = ModelLoadingContext.forModel(modelFile)) {
 *     ctx.setProgressCallback((loaded, total) -> {
 *         log.info("Loading: {}%", (loaded * 100) / total);
 *     });
 *     SameDiff model = SDZSerializer.load(modelFile, false, ctx);
 * }
 * }</pre>
 *
 * Adam Gibson
 * @see ModelSizeInfo
 * @see DeviceMemoryManager
 * @see TransferMetrics
 */
@Slf4j
public class ModelLoadingContext implements AutoCloseable {

    // Existing infrastructure references
    private final DeviceMemoryManager memoryManager;
    private final DeviceRoutingConfiguration routingConfig;
    private final TransferMetrics metrics;

    // Loading configuration
    @Getter
    private final DeviceDescriptor targetDevice;

    @Getter
    private final ModelSizeInfo sizeInfo;

    // Loading state tracking
    private final List<DataBuffer> pendingTransfers = new CopyOnWriteArrayList<>();
    private final AtomicLong bytesLoaded = new AtomicLong();
    private final AtomicLong bytesTransferred = new AtomicLong();
    private final AtomicInteger arraysLoaded = new AtomicInteger();
    private final AtomicInteger transfersScheduled = new AtomicInteger();
    private final AtomicInteger transfersCompleted = new AtomicInteger();

    // Batch transfer collection for native batched API
    private final List<OpaqueDataBuffer> batchedBuffers = new CopyOnWriteArrayList<>();

    // Background transfer thread pool
    private final ExecutorService transferExecutor;
    private final List<Future<?>> pendingFutures = new CopyOnWriteArrayList<>();

    // Progress callback
    private volatile BiConsumer<Long, Long> progressCallback;

    // Timing
    private final long startTimeNanos;
    private volatile long endTimeNanos;

    // Configuration options
    @Getter
    private final boolean asyncEnabled;
    @Getter
    private final int parallelTransfers;
    @Getter
    private final boolean useBatchedNativeTransfer;

    /**
     * Default number of parallel transfer threads.
     */
    public static final int DEFAULT_PARALLEL_TRANSFERS = 4;

    /**
     * Minimum array size (bytes) for async transfer consideration.
     */
    public static final long DEFAULT_ASYNC_THRESHOLD_BYTES = 64 * 1024; // 64KB

    private ModelLoadingContext(Builder builder) {
        this.memoryManager = DeviceMemoryManager.getInstance();
        this.routingConfig = DeviceRoutingConfiguration.current();
        this.metrics = TransferMetrics.getInstance();

        this.sizeInfo = builder.sizeInfo != null ? builder.sizeInfo : ModelSizeInfo.empty();
        this.asyncEnabled = builder.asyncEnabled;
        this.parallelTransfers = builder.parallelTransfers;
        this.useBatchedNativeTransfer = builder.useBatchedNativeTransfer;

        // Determine target device — must match the active backend
        DeviceType activeBackendType = Nd4j.getBackendDeviceType();
        if (builder.targetDevice != null) {
            this.targetDevice = builder.targetDevice;
        } else {
            // Use DeviceMemoryManager to select best device for the model size
            DeviceDescriptor selected = memoryManager.selectDeviceForAllocation(sizeInfo.getTotalBytes());
            // Validate: don't select a GPU device when running on CPU backend
            if (selected.getDeviceType() != activeBackendType && activeBackendType == DeviceType.CPU) {
                selected = DeviceDescriptor.cpu();
            }
            this.targetDevice = selected;
        }

        // Create thread pool if async is enabled
        if (asyncEnabled && parallelTransfers > 0) {
            ThreadFactory tf = r -> {
                Thread t = new Thread(r, "ModelLoadingContext-Transfer");
                t.setDaemon(true);
                return t;
            };
            this.transferExecutor = Executors.newFixedThreadPool(parallelTransfers, tf);
        } else {
            this.transferExecutor = null;
        }

        this.startTimeNanos = System.nanoTime();

        log.info("ModelLoadingContext initialized: target={}, totalSize={}, arrays={}, async={}, parallelTransfers={}",
                targetDevice.getDeviceId(),
                sizeInfo.toSummaryString(),
                sizeInfo.getArrayCount(),
                asyncEnabled,
                parallelTransfers);
    }

    /**
     * Create a loading context for a model file.
     * Analyzes the manifest to determine model size and selects the best device.
     *
     * @param modelFile the model file to analyze
     * @return configured ModelLoadingContext
     * @throws IOException if manifest analysis fails
     */
    public static ModelLoadingContext forModel(File modelFile) throws IOException {
        ModelSizeInfo sizeInfo = analyzeModelFile(modelFile);
        return builder()
                .sizeInfo(sizeInfo)
                .build();
    }

    /**
     * Create a loading context with a pre-analyzed manifest.
     *
     * @param manifest map of variable names to (offset, length) pairs
     * @return configured ModelLoadingContext
     */
    public static ModelLoadingContext forManifest(Map<String, Pair<Long, Long>> manifest) {
        ModelSizeInfo sizeInfo = ModelSizeInfo.fromManifest(manifest);
        return builder()
                .sizeInfo(sizeInfo)
                .build();
    }

    /**
     * Create a builder for custom configuration.
     *
     * @return new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Analyze a model file to extract size information.
     * This reads the manifest from the file without fully loading it.
     *
     * @param modelFile the model file to analyze
     * @return ModelSizeInfo with size statistics
     * @throws IOException if reading fails
     */
    private static ModelSizeInfo analyzeModelFile(File modelFile) throws IOException {
        if (!modelFile.exists()) {
            log.warn("Model file does not exist: {}. Returning empty ModelSizeInfo.", modelFile);
            return ModelSizeInfo.empty();
        }

        // For SDZ files, we need to extract and read the manifest
        // For SDNB files, we can read the manifest directly
        String filename = modelFile.getName().toLowerCase();

        if (filename.endsWith(".sdz")) {
            // For SDZ, we would need to peek inside the ZIP to read the manifest
            // For now, return an estimate based on file size
            log.debug("Estimating size for SDZ file: {}", modelFile.getName());
            return ModelSizeInfo.builder()
                    .totalBytes(modelFile.length())
                    .arrayCount(0) // Unknown
                    .largestArrayBytes(0) // Unknown
                    .arraySizes(Collections.emptyMap())
                    .build();
        } else if (filename.endsWith(".sdnb")) {
            // Read SDNB header to get manifest
            return analyzeSDNBFile(modelFile);
        }

        // Fallback: estimate based on file size
        return ModelSizeInfo.builder()
                .totalBytes(modelFile.length())
                .arrayCount(0)
                .largestArrayBytes(0)
                .arraySizes(Collections.emptyMap())
                .build();
    }

    /**
     * Analyze an SDNB file to extract manifest information.
     */
    private static ModelSizeInfo analyzeSDNBFile(File sdnbFile) throws IOException {
        // SDNB header format:
        // 4 bytes: magic "SDNB"
        // 4 bytes: version
        // 8 bytes: metadata length
        // 8 bytes: manifest length (number of entries)
        // Then manifest entries: varName -> (offset, length)

        try (RandomAccessFile raf = new RandomAccessFile(sdnbFile, "r");
             FileChannel channel = raf.getChannel()) {

            // Read header
            ByteBuffer header = ByteBuffer.allocate(32);
            header.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header);
            header.flip();

            // Check magic
            byte[] magic = new byte[4];
            header.get(magic);
            if (!new String(magic).equals("SDNB")) {
                log.warn("Not a valid SDNB file: {}. Magic bytes don't match.", sdnbFile.getName());
                return ModelSizeInfo.builder()
                        .totalBytes(sdnbFile.length())
                        .arrayCount(0)
                        .largestArrayBytes(0)
                        .arraySizes(Collections.emptyMap())
                        .build();
            }

            // For detailed manifest reading, we would need to fully parse the header
            // For now, return an estimate based on file size
            return ModelSizeInfo.builder()
                    .totalBytes(sdnbFile.length())
                    .arrayCount(0) // Would need full parse
                    .largestArrayBytes(0) // Would need full parse
                    .arraySizes(Collections.emptyMap())
                    .build();
        }
    }

    /**
     * Set a callback for progress updates during loading.
     * The callback receives (bytesLoaded, totalBytes).
     *
     * @param callback the progress callback
     */
    public void setProgressCallback(BiConsumer<Long, Long> callback) {
        this.progressCallback = callback;
    }

    /**
     * Called when an array has been loaded from disk.
     * Records the load and optionally schedules async transfer.
     *
     * @param array the loaded array
     */
    public void onArrayLoaded(INDArray array) {
        if (array == null || array.data() == null) {
            return;
        }

        long bytes = array.data().length() * array.data().getElementSize();
        bytesLoaded.addAndGet(bytes);
        arraysLoaded.incrementAndGet();

        // Notify progress
        if (progressCallback != null) {
            progressCallback.accept(bytesLoaded.get(), sizeInfo.getTotalBytes());
        }

        log.trace("Array loaded: {} bytes, total loaded: {} bytes", bytes, bytesLoaded.get());
    }

    /**
     * Schedule a background transfer for an array to the target device.
     * If async is disabled, performs synchronous transfer.
     *
     * @param array the array to transfer
     */
    public void scheduleTransfer(INDArray array) {
        if (array == null || array.data() == null) {
            return;
        }

        DataBuffer dataBuffer = array.data();
        long bytes = dataBuffer.length() * dataBuffer.getElementSize();

        // Check if async transfer is beneficial
        boolean useAsync = asyncEnabled &&
                transferExecutor != null &&
                bytes >= DEFAULT_ASYNC_THRESHOLD_BYTES;

        if (useAsync) {
            pendingTransfers.add(dataBuffer);
            transfersScheduled.incrementAndGet();

            // Check if we should use batched native transfer
            if (useBatchedNativeTransfer && dataBuffer.opaqueBuffer() != null) {
                batchedBuffers.add(dataBuffer.opaqueBuffer());
            } else {
                // Schedule individual async transfer
                Future<?> future = transferExecutor.submit(() -> {
                    try {
                        performTransfer(dataBuffer, bytes);
                    } catch (Exception e) {
                        log.error("Background transfer failed", e);
                    }
                });
                pendingFutures.add(future);
            }
        } else {
            // Synchronous transfer
            performTransfer(dataBuffer, bytes);
        }
    }

    /**
     * Perform the actual data transfer.
     */
    private void performTransfer(DataBuffer dataBuffer, long bytes) {
        long startNanos = System.nanoTime();

        try {
            if (dataBuffer instanceof HybridDataBuffer) {
                HybridDataBuffer hybrid = (HybridDataBuffer) dataBuffer;
                hybrid.ensureAvailableOn(targetDevice);
            } else if (dataBuffer.opaqueBuffer() != null) {
                // Use OpaqueDataBuffer sync
                dataBuffer.opaqueBuffer().syncToSpecial();
            }

            long elapsedNanos = System.nanoTime() - startNanos;
            bytesTransferred.addAndGet(bytes);
            transfersCompleted.incrementAndGet();

            // Record metrics
            metrics.recordTransfer(
                    DeviceDescriptor.cpu(),
                    targetDevice,
                    bytes,
                    elapsedNanos
            );

            log.trace("Transfer completed: {} bytes in {:.2f} ms to {}",
                    bytes, elapsedNanos / 1_000_000.0, targetDevice.getDeviceId());
        } catch (Exception e) {
            log.error("Transfer failed for buffer of {} bytes", bytes, e);
        }
    }

    /**
     * Execute batched native transfers for all collected buffers.
     * Attempts to use native CUDA stream-based batched transfer for optimal performance.
     * Falls back to parallel individual syncs via thread pool if native method unavailable.
     */
    public void executeBatchedTransfers() {
        if (batchedBuffers.isEmpty()) {
            return;
        }

        log.info("Executing batched transfer for {} buffers with {} parallel streams",
                batchedBuffers.size(), parallelTransfers);

        long startNanos = System.nanoTime();
        List<OpaqueDataBuffer> buffersToProcess = new ArrayList<>(batchedBuffers);
        batchedBuffers.clear();

        boolean nativeSuccess = false;

        // Try native batched transfer using CUDA streams
        try {
            nativeSuccess = executeNativeBatchedTransfer(buffersToProcess);
        } catch (Exception e) {
            log.debug("Native batched transfer not available, using thread pool fallback: {}", e.getMessage());
        }

        // Fallback to thread pool parallelism if native method failed
        if (!nativeSuccess) {
            executeThreadPoolTransfer(buffersToProcess);
        }

        // Calculate total bytes transferred (estimate based on element count)
        long totalBytes = 0;
        for (OpaqueDataBuffer buffer : buffersToProcess) {
            try {
                // Estimate: numElements * 4 bytes (assuming float32 average)
                totalBytes += buffer.numElements() * 4;
            } catch (Exception e) {
                // Ignore calculation errors
            }
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        bytesTransferred.addAndGet(totalBytes);
        transfersCompleted.addAndGet(buffersToProcess.size());

        // Record aggregate metrics
        metrics.recordTransfer(
                DeviceDescriptor.cpu(),
                targetDevice,
                totalBytes,
                elapsedNanos
        );

        log.info("Batched transfer completed: {} bytes across {} buffers in {:.2f} ms (native={})",
                totalBytes, buffersToProcess.size(), elapsedNanos / 1_000_000.0, nativeSuccess);
    }

    /**
     * Execute batched transfer using native CUDA streams via reflection.
     * Returns true if successful, false if native method not available.
     */
    private boolean executeNativeBatchedTransfer(List<OpaqueDataBuffer> buffers) {
        try {
            // Get the native ops implementation
            Object nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

            // Try to find and invoke the batchSyncToSpecialAsync method with PointerPointer
            // The generated signature is: batchSyncToSpecialAsync(PointerPointer, int, int)
            Class<?> pointerPointerClass = Class.forName("org.bytedeco.javacpp.PointerPointer");

            java.lang.reflect.Method method = nativeOps.getClass().getMethod(
                    "batchSyncToSpecialAsync", pointerPointerClass, int.class, int.class);

            // Create PointerPointer from the buffer list
            Object pointerPointer = pointerPointerClass.getConstructor(long.class).newInstance((long) buffers.size());

            // Put each buffer pointer into the PointerPointer
            java.lang.reflect.Method putMethod = pointerPointerClass.getMethod("put", long.class, org.bytedeco.javacpp.Pointer.class);
            for (int i = 0; i < buffers.size(); i++) {
                putMethod.invoke(pointerPointer, (long) i, buffers.get(i));
            }

            // Call the native method
            method.invoke(nativeOps, pointerPointer, buffers.size(), parallelTransfers);

            log.debug("Native batched transfer completed for {} buffers", buffers.size());
            return true;
        } catch (ClassNotFoundException e) {
            log.trace("PointerPointer class not found, native batch transfer not available");
            return false;
        } catch (NoSuchMethodException e) {
            log.trace("batchSyncToSpecialAsync method not found, native batch transfer not available");
            return false;
        } catch (Exception e) {
            log.debug("Native batched transfer failed: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Execute transfers using Java thread pool parallelism (fallback).
     */
    private void executeThreadPoolTransfer(List<OpaqueDataBuffer> buffers) {
        List<Future<?>> futures = new ArrayList<>();
        for (OpaqueDataBuffer buffer : buffers) {
            if (transferExecutor != null) {
                futures.add(transferExecutor.submit(() -> {
                    try {
                        buffer.syncToSpecial();
                    } catch (Exception e) {
                        log.error("Batch transfer failed for buffer", e);
                    }
                }));
            } else {
                // Fallback: synchronous execution
                try {
                    buffer.syncToSpecial();
                } catch (Exception e) {
                    log.error("Sync transfer failed for buffer", e);
                }
            }
        }

        // Wait for all parallel transfers
        for (Future<?> future : futures) {
            try {
                future.get(5, TimeUnit.MINUTES);
            } catch (Exception e) {
                log.error("Waiting for batch transfer failed", e);
            }
        }
    }

    /**
     * Wait for all background transfers to complete.
     */
    public void awaitTransfers() {
        // First, execute any batched transfers
        if (!batchedBuffers.isEmpty()) {
            executeBatchedTransfers();
        }

        // Then wait for individual async transfers
        if (transferExecutor != null && !pendingFutures.isEmpty()) {
            log.info("Waiting for {} background transfers to complete...", pendingFutures.size());

            for (Future<?> future : pendingFutures) {
                try {
                    future.get(30, TimeUnit.MINUTES);
                } catch (TimeoutException e) {
                    log.warn("Transfer timed out after 30 minutes");
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    log.warn("Transfer interrupted");
                } catch (ExecutionException e) {
                    log.error("Transfer execution failed", e.getCause());
                }
            }
            pendingFutures.clear();
        }
    }

    /**
     * Get loading statistics.
     *
     * @return map of statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("targetDevice", targetDevice.getDeviceId());
        stats.put("arraysLoaded", arraysLoaded.get());
        stats.put("bytesLoaded", bytesLoaded.get());
        stats.put("bytesTransferred", bytesTransferred.get());
        stats.put("transfersScheduled", transfersScheduled.get());
        stats.put("transfersCompleted", transfersCompleted.get());
        stats.put("pendingTransfers", pendingTransfers.size());

        long elapsed = endTimeNanos > 0 ? endTimeNanos - startTimeNanos : System.nanoTime() - startTimeNanos;
        stats.put("elapsedTimeMs", elapsed / 1_000_000.0);

        if (elapsed > 0 && bytesLoaded.get() > 0) {
            double loadBandwidthGBps = (bytesLoaded.get() / 1e9) / (elapsed / 1e9);
            stats.put("loadBandwidthGBps", loadBandwidthGBps);
        }

        return stats;
    }

    /**
     * Log a summary of the loading operation.
     */
    public void logSummary() {
        Map<String, Object> stats = getStatistics();

        log.info("========== MODEL LOADING SUMMARY ==========");
        log.info("Target device:      {}", stats.get("targetDevice"));
        log.info("Arrays loaded:      {}", stats.get("arraysLoaded"));
        log.info("Bytes loaded:       {} MB", ((Long) stats.get("bytesLoaded")) / (1024.0 * 1024.0));
        log.info("Bytes transferred:  {} MB", ((Long) stats.get("bytesTransferred")) / (1024.0 * 1024.0));
        log.info("Elapsed time:       {:.2f} ms", stats.get("elapsedTimeMs"));

        if (stats.containsKey("loadBandwidthGBps")) {
            log.info("Load bandwidth:     {:.2f} GB/s", stats.get("loadBandwidthGBps"));
        }

        log.info("============================================");
    }

    @Override
    public void close() {
        endTimeNanos = System.nanoTime();

        // Wait for pending transfers
        awaitTransfers();

        // Shutdown executor
        if (transferExecutor != null) {
            transferExecutor.shutdown();
            try {
                if (!transferExecutor.awaitTermination(1, TimeUnit.MINUTES)) {
                    transferExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                transferExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        // Log summary
        logSummary();

        // Also log transfer metrics summary
        metrics.logSummary();

        log.info("ModelLoadingContext closed: {} bytes loaded, {} bytes transferred to {}",
                bytesLoaded.get(), bytesTransferred.get(), targetDevice.getDeviceId());
    }

    /**
     * Builder for ModelLoadingContext.
     */
    public static class Builder {
        private DeviceDescriptor targetDevice;
        private ModelSizeInfo sizeInfo;
        private boolean asyncEnabled = true;
        private int parallelTransfers = DEFAULT_PARALLEL_TRANSFERS;
        private boolean useBatchedNativeTransfer = true;

        /**
         * Set the target device for loading.
         * If not set, the best device is selected automatically.
         *
         * @param device the target device
         * @return this builder
         */
        public Builder targetDevice(DeviceDescriptor device) {
            this.targetDevice = device;
            return this;
        }

        /**
         * Set the model size information.
         *
         * @param sizeInfo pre-analyzed model size info
         * @return this builder
         */
        public Builder sizeInfo(ModelSizeInfo sizeInfo) {
            this.sizeInfo = sizeInfo;
            return this;
        }

        /**
         * Enable or disable async transfers.
         *
         * @param enabled true to enable async transfers
         * @return this builder
         */
        public Builder asyncEnabled(boolean enabled) {
            this.asyncEnabled = enabled;
            return this;
        }

        /**
         * Set the number of parallel transfer threads.
         *
         * @param parallelTransfers number of threads
         * @return this builder
         */
        public Builder parallelTransfers(int parallelTransfers) {
            this.parallelTransfers = parallelTransfers;
            return this;
        }

        /**
         * Enable or disable batched native transfers.
         *
         * @param enabled true to use native batched API
         * @return this builder
         */
        public Builder useBatchedNativeTransfer(boolean enabled) {
            this.useBatchedNativeTransfer = enabled;
            return this;
        }

        /**
         * Build the ModelLoadingContext.
         *
         * @return configured context
         */
        public ModelLoadingContext build() {
            return new ModelLoadingContext(this);
        }
    }
}
