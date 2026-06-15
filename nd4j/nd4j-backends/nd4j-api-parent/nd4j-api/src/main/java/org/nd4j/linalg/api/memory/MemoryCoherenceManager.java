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

package org.nd4j.linalg.api.memory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.HybridDataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Manages memory coherence across devices using an MSI-like protocol.
 * Tracks buffer states and coordinates data transfers to maintain consistency.
 *
 * <p>The coherence protocol states are:</p>
 * <ul>
 *   <li>EXCLUSIVE - Only copy, can read/write freely</li>
 *   <li>SHARED - Valid copy shared with others, read-only</li>
 *   <li>MODIFIED - Dirty copy, must write back before others can read</li>
 *   <li>INVALID - Stale copy, must fetch before use</li>
 * </ul>
 */
@Slf4j
public class MemoryCoherenceManager {

    private static volatile MemoryCoherenceManager instance;
    private static final Object INSTANCE_LOCK = new Object();

    // Track all hybrid buffers and their states
    private final Map<Long, BufferCoherenceInfo> bufferStates = new ConcurrentHashMap<>();

    // Transfer queue for async operations
    private final ExecutorService transferExecutor;
    private final BlockingQueue<TransferRequest> transferQueue;

    // Statistics
    private final AtomicLong totalTransfers = new AtomicLong(0);
    private final AtomicLong totalBytesTransferred = new AtomicLong(0);
    private final AtomicLong coherenceViolations = new AtomicLong(0);

    private MemoryCoherenceManager() {
        this.transferQueue = new LinkedBlockingQueue<>();
        this.transferExecutor = Executors.newFixedThreadPool(
                Math.max(2, Runtime.getRuntime().availableProcessors() / 2),
                r -> {
                    Thread t = new Thread(r, "memory-coherence-transfer");
                    t.setDaemon(true);
                    return t;
                }
        );
    }

    /**
     * Get the singleton instance
     * @return the coherence manager instance
     */
    public static MemoryCoherenceManager getInstance() {
        if (instance == null) {
            synchronized (INSTANCE_LOCK) {
                if (instance == null) {
                    instance = new MemoryCoherenceManager();
                }
            }
        }
        return instance;
    }

    /**
     * Register a buffer for coherence tracking
     * @param bufferId unique buffer identifier
     * @param ownerDevice initial owner device
     */
    public void registerBuffer(long bufferId, DeviceDescriptor ownerDevice) {
        bufferStates.put(bufferId, new BufferCoherenceInfo(bufferId, ownerDevice));
        log.debug("Registered buffer {} on device {}", bufferId, ownerDevice.getDeviceId());
    }

    /**
     * Unregister a buffer (on deallocation)
     * @param bufferId the buffer to unregister
     */
    public void unregisterBuffer(long bufferId) {
        bufferStates.remove(bufferId);
    }

    /**
     * Request read access to a buffer on a device
     * @param bufferId the buffer ID
     * @param device the device requesting access
     * @return the memory location for reading
     */
    public MemoryLocation requestRead(long bufferId, DeviceDescriptor device) {
        BufferCoherenceInfo info = bufferStates.get(bufferId);
        if (info == null) {
            throw new IllegalStateException("Buffer not registered: " + bufferId);
        }

        synchronized (info) {
            MemoryLocation location = info.getLocation(device);

            // If already valid on this device, just return
            if (location != null && location.isValid()) {
                location.touch();
                return location;
            }

            // Need to fetch from owner
            DeviceDescriptor owner = info.getOwner();
            if (owner == null) {
                throw new IllegalStateException("Buffer has no owner: " + bufferId);
            }

            MemoryLocation ownerLocation = info.getLocation(owner);
            if (ownerLocation == null || !ownerLocation.isValid()) {
                throw new IllegalStateException("Owner has no valid data: " + bufferId);
            }

            // Create new location on target device
            location = transferData(ownerLocation, device, false);

            // Update states
            ownerLocation.markShared();  // Owner becomes SHARED
            location.setState(CoherenceState.SHARED);  // New copy is SHARED
            info.addLocation(device, location);

            return location;
        }
    }

    /**
     * Request write access to a buffer on a device
     * @param bufferId the buffer ID
     * @param device the device requesting access
     * @return the memory location for writing
     */
    public MemoryLocation requestWrite(long bufferId, DeviceDescriptor device) {
        BufferCoherenceInfo info = bufferStates.get(bufferId);
        if (info == null) {
            throw new IllegalStateException("Buffer not registered: " + bufferId);
        }

        synchronized (info) {
            MemoryLocation location = info.getLocation(device);

            // If already owner with exclusive access, return
            if (location != null && location.canWrite()) {
                location.touch();
                return location;
            }

            // Need to get exclusive access
            DeviceDescriptor currentOwner = info.getOwner();

            if (currentOwner != null && !currentOwner.equals(device)) {
                // Transfer from current owner
                MemoryLocation ownerLocation = info.getLocation(currentOwner);
                if (ownerLocation != null && ownerLocation.isValid()) {
                    location = transferData(ownerLocation, device, true);
                }
            }

            // Invalidate all other copies
            for (Map.Entry<String, MemoryLocation> entry : info.getAllLocations().entrySet()) {
                if (!entry.getKey().equals(device.getDeviceId())) {
                    entry.getValue().invalidate();
                }
            }

            // Set this as new owner with EXCLUSIVE state
            if (location == null) {
                // Need to create new allocation (shouldn't normally happen)
                throw new IllegalStateException("No valid source for write transfer");
            }
            location.setState(CoherenceState.EXCLUSIVE);
            info.setOwner(device);
            info.addLocation(device, location);

            return location;
        }
    }

    /**
     * Notify that a buffer was modified on a device
     * @param bufferId the buffer ID
     * @param device the device that modified the data
     */
    public void notifyModified(long bufferId, DeviceDescriptor device) {
        BufferCoherenceInfo info = bufferStates.get(bufferId);
        if (info == null) return;

        synchronized (info) {
            MemoryLocation location = info.getLocation(device);
            if (location != null) {
                location.markModified();
                info.setOwner(device);

                // Invalidate all other copies
                for (Map.Entry<String, MemoryLocation> entry : info.getAllLocations().entrySet()) {
                    if (!entry.getKey().equals(device.getDeviceId())) {
                        entry.getValue().invalidate();
                    }
                }
            }
        }
    }

    /**
     * Transfer data between devices
     * @param source source location
     * @param targetDevice target device
     * @param exclusive whether to transfer ownership
     * @return new memory location on target
     */
    private MemoryLocation transferData(MemoryLocation source, DeviceDescriptor targetDevice, boolean exclusive) {
        // This is a placeholder - actual transfer implementation depends on backend
        totalTransfers.incrementAndGet();
        totalBytesTransferred.addAndGet(source.getSize());

        // Create new location (in real implementation, this would allocate and copy)
        MemoryLocation target = new MemoryLocation(
                targetDevice,
                0,  // Address would be allocated by backend
                source.getSize(),
                exclusive ? CoherenceState.EXCLUSIVE : CoherenceState.SHARED,
                source.getVersion()
        );

        log.debug("Transfer {} bytes from {} to {}", source.getSize(),
                source.getDevice().getDeviceId(), targetDevice.getDeviceId());

        return target;
    }

    /**
     * Queue an async transfer
     * @param source source location
     * @param targetDevice target device
     * @return future for the transfer
     */
    public Future<MemoryLocation> queueTransfer(MemoryLocation source, DeviceDescriptor targetDevice) {
        CompletableFuture<MemoryLocation> future = new CompletableFuture<>();

        transferExecutor.submit(() -> {
            try {
                MemoryLocation result = transferData(source, targetDevice, false);
                future.complete(result);
            } catch (Exception e) {
                future.completeExceptionally(e);
            }
        });

        return future;
    }

    /**
     * Get coherence statistics
     * @return map of statistic name to value
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("trackedBuffers", bufferStates.size());
        stats.put("totalTransfers", totalTransfers.get());
        stats.put("totalBytesTransferred", totalBytesTransferred.get());
        stats.put("coherenceViolations", coherenceViolations.get());
        return stats;
    }

    /**
     * Shutdown the coherence manager
     */
    public void shutdown() {
        transferExecutor.shutdown();
        try {
            if (!transferExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                transferExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            transferExecutor.shutdownNow();
        }
    }

    /**
     * Internal class to track coherence info for a single buffer
     */
    private static class BufferCoherenceInfo {
        private final long bufferId;
        private volatile DeviceDescriptor owner;
        private final Map<String, MemoryLocation> locations = new ConcurrentHashMap<>();

        BufferCoherenceInfo(long bufferId, DeviceDescriptor initialOwner) {
            this.bufferId = bufferId;
            this.owner = initialOwner;
        }

        DeviceDescriptor getOwner() {
            return owner;
        }

        void setOwner(DeviceDescriptor device) {
            this.owner = device;
        }

        MemoryLocation getLocation(DeviceDescriptor device) {
            return locations.get(device.getDeviceId());
        }

        void addLocation(DeviceDescriptor device, MemoryLocation location) {
            locations.put(device.getDeviceId(), location);
        }

        Map<String, MemoryLocation> getAllLocations() {
            return Collections.unmodifiableMap(locations);
        }
    }

    /**
     * Internal class for async transfer requests
     */
    private static class TransferRequest {
        final MemoryLocation source;
        final DeviceDescriptor target;
        final CompletableFuture<MemoryLocation> future;

        TransferRequest(MemoryLocation source, DeviceDescriptor target) {
            this.source = source;
            this.target = target;
            this.future = new CompletableFuture<>();
        }
    }
}
