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

package org.nd4j.autodiff.samediff.training;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.GradientCheckpointConfig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages async offloading of gradient checkpoints to host (CPU) memory.
 *
 * <p>During the <b>forward pass</b>, checkpoint activations are copied to pinned
 * (page-locked) host memory via non-blocking D2H transfers so the GPU can
 * overlap compute with data movement.
 *
 * <p>During the <b>backward pass</b>, activations are prefetched back to the
 * device ({@link #prefetchForBackward}) a configurable number of layers ahead of
 * when they are actually consumed ({@link #getCheckpoint}).
 *
 * <p>Example:
 * <pre>
 * GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload(2);
 * CheckpointOffloadManager mgr = new CheckpointOffloadManager(cfg);
 *
 * // --- forward pass ---
 * for (int layer = 0; layer &lt; numLayers; layer++) {
 *     INDArray activation = runLayer(layer);
 *     mgr.offloadCheckpoint("layer_" + layer, activation);
 * }
 *
 * // --- backward pass ---
 * for (int layer = numLayers - 1; layer &gt;= 0; layer--) {
 *     mgr.prefetchForBackward(layer, numLayers);
 *     INDArray activation = mgr.getCheckpoint("layer_" + layer);
 *     runBackward(layer, activation);
 * }
 *
 * mgr.releaseAll();
 * </pre>
 *
 * <p>This implementation is backend-agnostic: on CPU the offload is a simple
 * {@code dup()}, and on CUDA it uses the {@code checkpoint_offload_d2h} /
 * {@code checkpoint_prefetch_h2d} ops which trigger async DMA when pinned
 * memory is enabled.
 *
 * @author Adam Gibson
 */
@Slf4j
public class CheckpointOffloadManager {

    /**
     * The config this manager was created with.
     */
    @Getter
    private final GradientCheckpointConfig config;

    /**
     * Host-resident copies of checkpointed activations.
     * Key: variable name.  Value: copy on CPU (host).
     */
    private final Map<String, INDArray> hostCheckpoints;

    /**
     * Device-resident arrays that have been prefetched back from host.
     * Key: variable name.  Value: copy on device (GPU).
     */
    private final Map<String, INDArray> deviceCheckpoints;

    /**
     * Ordered list of variable names in forward-pass order.
     * Used by {@link #prefetchForBackward} to determine which checkpoints
     * to prefetch based on the current backward layer index.
     */
    private final List<String> forwardOrder;

    /** Cumulative bytes currently held in host memory. */
    private long hostMemoryBytes;

    /**
     * Construct a CheckpointOffloadManager using the given configuration.
     *
     * @param config gradient checkpoint configuration; must have strategy
     *               {@code OFFLOAD_HOST} or {@code OFFLOAD_HOST_ASYNC}
     */
    public CheckpointOffloadManager(@NonNull GradientCheckpointConfig config) {
        this.config = config;
        this.hostCheckpoints = new LinkedHashMap<>();
        this.deviceCheckpoints = new ConcurrentHashMap<>();
        this.forwardOrder = new ArrayList<>();
        this.hostMemoryBytes = 0L;
    }

    // -------------------------------------------------------------------------
    // Forward pass API
    // -------------------------------------------------------------------------

    /**
     * Schedule an async D2H copy of {@code activation} to host memory.
     *
     * <p>If the host memory budget ({@link GradientCheckpointConfig#getMaxHostMemoryMB()})
     * would be exceeded the activation is skipped and a warning is logged.
     *
     * @param variableName logical name of the variable (used as key)
     * @param activation   device or host array to offload
     */
    public void offloadCheckpoint(@NonNull String variableName, @NonNull INDArray activation) {
        long activationBytes = activation.length() * activation.dataType().width();

        // Budget check
        long maxBytes = config.getMaxHostMemoryMB() * 1024L * 1024L;
        if (maxBytes > 0 && hostMemoryBytes + activationBytes > maxBytes) {
            log.warn("CheckpointOffloadManager: host memory budget exceeded, skipping offload for '{}'", variableName);
            return;
        }

        INDArray hostCopy = execD2H(variableName, activation);
        hostCheckpoints.put(variableName, hostCopy);
        forwardOrder.add(variableName);
        hostMemoryBytes += activationBytes;

        log.debug("CheckpointOffloadManager: offloaded '{}' ({} bytes, total host={} MB)",
                variableName, activationBytes, hostMemoryBytes / (1024 * 1024));
    }

    // -------------------------------------------------------------------------
    // Backward pass API
    // -------------------------------------------------------------------------

    /**
     * Retrieve the checkpointed activation for {@code variableName}.
     *
     * <p>If a prefetched device copy already exists it is returned immediately
     * (and removed from the prefetch cache). Otherwise a synchronous H2D copy
     * is performed.
     *
     * @param variableName the variable name passed to {@link #offloadCheckpoint}
     * @return device-resident array with the checkpointed data
     * @throws IllegalArgumentException if no checkpoint exists for the variable
     */
    public INDArray getCheckpoint(@NonNull String variableName) {
        // 1. Fast path: already prefetched
        INDArray prefetched = deviceCheckpoints.remove(variableName);
        if (prefetched != null) {
            log.debug("CheckpointOffloadManager: prefetch hit for '{}'", variableName);
            return prefetched;
        }

        // 2. Synchronous H2D fallback
        INDArray hostCopy = hostCheckpoints.get(variableName);
        if (hostCopy == null) {
            throw new IllegalArgumentException(
                    "CheckpointOffloadManager: no checkpoint found for '" + variableName + "'");
        }
        log.debug("CheckpointOffloadManager: sync H2D for '{}'", variableName);
        return execH2D(variableName, hostCopy);
    }

    /**
     * Prefetch checkpoints for layers that will be needed within
     * {@link GradientCheckpointConfig#getPrefetchDistance()} steps from the current layer.
     *
     * <p>Call this at the start of each backward layer before calling
     * {@link #getCheckpoint} for that layer.
     *
     * @param currentLayer zero-based backward layer index (decreasing during backward)
     * @param totalLayers  total number of layers in the model
     */
    public void prefetchForBackward(int currentLayer, int totalLayers) {
        if (!config.isAsyncOffload()) {
            return;  // sync offload path: no prefetch needed
        }
        int dist = config.getPrefetchDistance();
        // During backward we iterate layers from totalLayers-1 down to 0.
        // "currentLayer" is the layer we are ABOUT to process.
        // Prefetch for layers currentLayer - 1 ... currentLayer - dist.
        for (int d = 1; d <= dist; d++) {
            int target = currentLayer - d;
            if (target < 0 || target >= forwardOrder.size()) continue;
            String varName = forwardOrder.get(target);
            if (hostCheckpoints.containsKey(varName)
                    && !deviceCheckpoints.containsKey(varName)) {
                log.debug("CheckpointOffloadManager: prefetching '{}' (d={})", varName, d);
                INDArray deviceCopy = execH2D(varName, hostCheckpoints.get(varName));
                deviceCheckpoints.put(varName, deviceCopy);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /**
     * Release all host and device arrays held by this manager.
     * Must be called after each training step to avoid memory leaks.
     */
    public void releaseAll() {
        for (INDArray arr : hostCheckpoints.values()) {
            closeQuietly(arr);
        }
        hostCheckpoints.clear();

        for (INDArray arr : deviceCheckpoints.values()) {
            closeQuietly(arr);
        }
        deviceCheckpoints.clear();

        forwardOrder.clear();
        hostMemoryBytes = 0L;
    }

    /**
     * Return the number of activations currently held in host memory.
     */
    public int getHostCheckpointCount() {
        return hostCheckpoints.size();
    }

    /**
     * Return the total bytes currently occupying host memory.
     */
    public long getHostMemoryBytes() {
        return hostMemoryBytes;
    }

    /**
     * Return the number of prefetched device tensors not yet consumed.
     */
    public int getPrefetchedCount() {
        return deviceCheckpoints.size();
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Execute a device-to-host transfer for {@code src}.
     *
     * When the {@code checkpoint_offload_d2h} op is available and the backend
     * is CUDA, this will be an async DMA transfer to pinned memory.
     * On CPU or when the op is not registered, falls back to {@code src.dup()}.
     */
    private INDArray execD2H(String variableName, INDArray src) {
        try {
            int usePinned = config.isPinHostMemory() ? 1 : 0;
            INDArray[] out = Nd4j.exec(
                    org.nd4j.linalg.api.ops.DynamicCustomOp.builder("checkpoint_offload_d2h")
                            .addInputs(src)
                            .addIntegerArguments(usePinned, 0)
                            .build());
            return out[0];
        } catch (Exception e) {
            // Op not registered on this backend (e.g., pure CPU without CUDA) — fall back
            log.trace("checkpoint_offload_d2h not available for '{}', using dup(): {}", variableName, e.getMessage());
            return src.dup();
        }
    }

    /**
     * Execute a host-to-device transfer for {@code src}.
     *
     * When the {@code checkpoint_prefetch_h2d} op is available and the backend
     * is CUDA, this will be an async DMA transfer from pinned memory.
     * On CPU or when the op is not registered, falls back to {@code src.dup()}.
     */
    private INDArray execH2D(String variableName, INDArray src) {
        try {
            INDArray[] out = Nd4j.exec(
                    org.nd4j.linalg.api.ops.DynamicCustomOp.builder("checkpoint_prefetch_h2d")
                            .addInputs(src)
                            .addIntegerArguments(0)
                            .build());
            return out[0];
        } catch (Exception e) {
            // Op not registered on this backend — fall back
            log.trace("checkpoint_prefetch_h2d not available for '{}', using dup(): {}", variableName, e.getMessage());
            return src.dup();
        }
    }

    private static void closeQuietly(INDArray arr) {
        if (arr != null && !arr.wasClosed()) {
            try {
                arr.close();
            } catch (Exception e) {
                log.warn("CheckpointOffloadManager: error closing array", e);
            }
        }
    }
}
