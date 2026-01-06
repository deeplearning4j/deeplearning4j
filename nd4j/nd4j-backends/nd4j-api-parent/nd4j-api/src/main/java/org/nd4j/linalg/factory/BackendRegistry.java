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

package org.nd4j.linalg.factory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.memory.MultiBackendWorkspaceManager;
import org.nd4j.linalg.api.memory.conf.DeviceAwareWorkspaceConfiguration;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

/**
 * Central registry for managing backends and devices in ND4J.
 * Provides thread-safe lazy initialization and device discovery.
 */
@Slf4j
public class BackendRegistry {

    private static volatile BackendRegistry instance;
    private static final Object INSTANCE_LOCK = new Object();

    // Backend storage
    private final Map<String, Nd4jBackend> backends = new ConcurrentHashMap<>();
    private final Map<String, OpExecutioner> executioners = new ConcurrentHashMap<>();

    // Device storage
    private final Map<String, DeviceDescriptor> devicesById = new ConcurrentHashMap<>();
    private final Map<DeviceType, List<DeviceDescriptor>> devicesByType = new ConcurrentHashMap<>();

    // Initialization state
    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    // Default devices
    private volatile DeviceDescriptor defaultCpuDevice;
    private volatile DeviceDescriptor defaultGpuDevice;

    private BackendRegistry() {
        // Private constructor for singleton
    }

    /**
     * Get the singleton instance
     * @return the registry instance
     */
    public static BackendRegistry getInstance() {
        if (instance == null) {
            synchronized (INSTANCE_LOCK) {
                if (instance == null) {
                    instance = new BackendRegistry();
                }
            }
        }
        return instance;
    }

    /**
     * Initialize the registry by discovering backends and devices
     */
    public void initialize() {
        if (initialized.get()) {
            return;
        }

        lock.writeLock().lock();
        try {
            if (initialized.get()) {
                return;
            }

            log.debug("Initializing BackendRegistry");
            discoverBackends();
            discoverDevices();
            initialized.set(true);
            log.info("BackendRegistry initialized with {} backends and {} devices",
                    backends.size(), devicesById.size());

        } finally {
            lock.writeLock().unlock();
        }
    }

    private void discoverBackends() {
        ServiceLoader<Nd4jBackend> loader = ND4JClassLoading.loadService(Nd4jBackend.class);

        for (Nd4jBackend backend : loader) {
            try {
                if (backend.isAvailable()) {
                    String backendId = backend.getBackendId();
                    backends.put(backendId, backend);
                    backend.initialize();
                    log.debug("Registered backend: {}", backendId);
                }
            } catch (Exception e) {
                log.warn("Failed to initialize backend: {}", backend.getClass().getName(), e);
            }
        }
    }

    private void discoverDevices() {
        for (Map.Entry<String, Nd4jBackend> entry : backends.entrySet()) {
            try {
                List<DeviceDescriptor> devices = entry.getValue().discoverDevices();
                for (DeviceDescriptor device : devices) {
                    registerDevice(device);
                }
            } catch (Exception e) {
                log.warn("Failed to discover devices for backend: {}", entry.getKey(), e);
            }
        }
    }

    private void registerDevice(DeviceDescriptor device) {
        devicesById.put(device.getDeviceId(), device);

        devicesByType.computeIfAbsent(device.getDeviceType(), k -> new ArrayList<>())
                .add(device);

        // Track defaults
        if (device.isDefault()) {
            if (device.getDeviceType() == DeviceType.CPU && defaultCpuDevice == null) {
                defaultCpuDevice = device;
            } else if (device.getDeviceType().isGpu() && defaultGpuDevice == null) {
                defaultGpuDevice = device;
            }
        }

        log.debug("Registered device: {}", device.getDeviceId());
    }

    /**
     * Get all registered backends
     * @return map of backend ID to backend
     */
    public Map<String, Nd4jBackend> getBackends() {
        ensureInitialized();
        return Collections.unmodifiableMap(backends);
    }

    /**
     * Get a backend by ID
     * @param backendId the backend ID
     * @return the backend, or null if not found
     */
    public Nd4jBackend getBackend(String backendId) {
        ensureInitialized();
        return backends.get(backendId);
    }

    /**
     * Get all devices
     * @return list of all devices
     */
    public List<DeviceDescriptor> getAllDevices() {
        ensureInitialized();
        return new ArrayList<>(devicesById.values());
    }

    /**
     * Get a device by ID
     * @param deviceId the device ID
     * @return the device, or null if not found
     */
    public DeviceDescriptor getDevice(String deviceId) {
        ensureInitialized();
        return devicesById.get(deviceId);
    }

    /**
     * Get all devices of a specific type
     * @param type the device type
     * @return list of devices
     */
    public List<DeviceDescriptor> getDevices(DeviceType type) {
        ensureInitialized();
        List<DeviceDescriptor> devices = devicesByType.get(type);
        return devices != null ? new ArrayList<>(devices) : Collections.emptyList();
    }

    /**
     * Get the default CPU device
     * @return the default CPU device, or null if none
     */
    public DeviceDescriptor getDefaultCpuDevice() {
        ensureInitialized();
        return defaultCpuDevice;
    }

    /**
     * Get the default GPU device
     * @return the default GPU device, or null if none
     */
    public DeviceDescriptor getDefaultGpuDevice() {
        ensureInitialized();
        return defaultGpuDevice;
    }

    /**
     * Get the default device for array creation.
     * Returns the first available GPU if present, otherwise CPU.
     * This follows the convention of frameworks like PyTorch/JAX that prefer
     * accelerators when available.
     *
     * @return the default device (GPU if available, else CPU)
     */
    public DeviceDescriptor getDefaultDevice() {
        ensureInitialized();
        // Prefer GPU if available (like PyTorch/JAX)
        if (defaultGpuDevice != null) {
            return defaultGpuDevice;
        }
        return defaultCpuDevice;
    }

    /**
     * Check if any GPU devices are available
     * @return true if GPUs are available
     */
    public boolean hasGpu() {
        ensureInitialized();
        return defaultGpuDevice != null ||
                devicesByType.entrySet().stream()
                        .anyMatch(e -> e.getKey().isGpu() && !e.getValue().isEmpty());
    }

    /**
     * Get the first available GPU
     * @return the first GPU, or null if none
     */
    public DeviceDescriptor getFirstGpu() {
        ensureInitialized();
        if (defaultGpuDevice != null) {
            return defaultGpuDevice;
        }
        for (Map.Entry<DeviceType, List<DeviceDescriptor>> entry : devicesByType.entrySet()) {
            if (entry.getKey().isGpu() && !entry.getValue().isEmpty()) {
                return entry.getValue().get(0);
            }
        }
        return null;
    }

    /**
     * Get all accelerator devices (non-CPU)
     * @return list of accelerator devices
     */
    public List<DeviceDescriptor> getAccelerators() {
        ensureInitialized();
        return devicesById.values().stream()
                .filter(d -> d.getDeviceType().isAccelerator())
                .collect(Collectors.toList());
    }

    /**
     * Get or create an executioner for a backend
     * @param backendId the backend ID
     * @return the executioner
     */
    public OpExecutioner getExecutioner(String backendId) {
        ensureInitialized();
        return executioners.computeIfAbsent(backendId, id -> {
            Nd4jBackend backend = backends.get(id);
            if (backend != null) {
                return backend.createExecutioner();
            }
            return null;
        });
    }

    /**
     * Get the executioner for a device
     * @param device the device
     * @return the executioner
     */
    public OpExecutioner getExecutioner(DeviceDescriptor device) {
        return getExecutioner(device.getBackendId());
    }

    /**
     * Check if the registry is initialized
     * @return true if initialized
     */
    public boolean isInitialized() {
        return initialized.get();
    }

    private void ensureInitialized() {
        if (!initialized.get()) {
            initialize();
        }
    }

    /**
     * Reset the registry (mainly for testing)
     */
    public void reset() {
        lock.writeLock().lock();
        try {
            backends.clear();
            executioners.clear();
            devicesById.clear();
            devicesByType.clear();
            defaultCpuDevice = null;
            defaultGpuDevice = null;
            initialized.set(false);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Manually register a device (for testing or special cases)
     * @param device the device to register
     */
    public void registerDeviceManually(DeviceDescriptor device) {
        lock.writeLock().lock();
        try {
            registerDevice(device);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Get device count by type
     * @return map of device type to count
     */
    public Map<DeviceType, Integer> getDeviceCounts() {
        ensureInitialized();
        Map<DeviceType, Integer> counts = new EnumMap<>(DeviceType.class);
        for (Map.Entry<DeviceType, List<DeviceDescriptor>> entry : devicesByType.entrySet()) {
            counts.put(entry.getKey(), entry.getValue().size());
        }
        return counts;
    }

    // ========================
    // Workspace Support
    // ========================

    /**
     * Get the multi-backend workspace manager
     * @return the workspace manager
     */
    public MultiBackendWorkspaceManager getWorkspaceManager() {
        return MultiBackendWorkspaceManager.getInstance();
    }

    /**
     * Create a multi-backend workspace with the given configuration
     * @param configuration the workspace configuration
     * @param id the workspace ID
     * @return the created workspace
     */
    public MultiBackendWorkspace createWorkspace(DeviceAwareWorkspaceConfiguration configuration, String id) {
        ensureInitialized();
        return getWorkspaceManager().createWorkspace(configuration, id);
    }

    /**
     * Create a multi-backend workspace on a specific device
     * @param configuration the workspace configuration
     * @param id the workspace ID
     * @param device the target device
     * @return the created workspace
     */
    public MultiBackendWorkspace createWorkspaceOnDevice(DeviceAwareWorkspaceConfiguration configuration,
                                                          String id,
                                                          DeviceDescriptor device) {
        ensureInitialized();
        return getWorkspaceManager().createWorkspaceOnDevice(configuration, id, device);
    }

    /**
     * Get or create a multi-backend workspace
     * @param configuration the workspace configuration
     * @param id the workspace ID
     * @return the workspace
     */
    public MultiBackendWorkspace getOrCreateWorkspace(DeviceAwareWorkspaceConfiguration configuration, String id) {
        ensureInitialized();
        return getWorkspaceManager().getOrCreateWorkspace(configuration, id);
    }

    /**
     * Get and activate a multi-backend workspace (enter scope)
     * @param configuration the workspace configuration
     * @param id the workspace ID
     * @return the activated workspace
     */
    public MultiBackendWorkspace getAndActivateWorkspace(DeviceAwareWorkspaceConfiguration configuration, String id) {
        ensureInitialized();
        return getWorkspaceManager().getAndActivateWorkspace(configuration, id);
    }

    /**
     * Get an existing multi-backend workspace
     * @param id the workspace ID
     * @return the workspace, or null if not found
     */
    public MultiBackendWorkspace getWorkspace(String id) {
        return getWorkspaceManager().getWorkspace(id);
    }

    /**
     * Destroy a multi-backend workspace
     * @param workspace the workspace to destroy
     */
    public void destroyWorkspace(MultiBackendWorkspace workspace) {
        getWorkspaceManager().destroyWorkspace(workspace);
    }

    /**
     * Get total memory usage across all multi-backend workspaces for the current thread
     * @return total memory in bytes
     */
    public long getTotalWorkspaceMemory() {
        return getWorkspaceManager().getTotalMemoryUsage();
    }

    /**
     * Get memory usage per device across all multi-backend workspaces
     * @return map of device ID to total memory
     */
    public Map<String, Long> getWorkspaceMemoryPerDevice() {
        return getWorkspaceManager().getMemoryUsagePerDevice();
    }

    /**
     * Check if a multi-backend workspace exists
     * @param id the workspace ID
     * @return true if exists
     */
    public boolean workspaceExists(String id) {
        return getWorkspaceManager().workspaceExists(id);
    }

    /**
     * Get all workspace IDs for the current thread
     * @return set of workspace IDs
     */
    public Set<String> getWorkspaceIds() {
        return getWorkspaceManager().getWorkspaceIdsForCurrentThread();
    }

    /**
     * Create a GPU-preferred workspace configuration
     * @param initialSize initial workspace size
     * @return the configuration
     */
    public DeviceAwareWorkspaceConfiguration gpuPreferredWorkspaceConfig(long initialSize) {
        return DeviceAwareWorkspaceConfiguration.gpuPreferred(initialSize);
    }

    /**
     * Create a CPU-only workspace configuration
     * @param initialSize initial workspace size
     * @return the configuration
     */
    public DeviceAwareWorkspaceConfiguration cpuOnlyWorkspaceConfig(long initialSize) {
        return DeviceAwareWorkspaceConfiguration.cpuOnly(initialSize);
    }

    /**
     * Create a mirrored workspace configuration (data on both CPU and GPU)
     * @param initialSize initial workspace size
     * @return the configuration
     */
    public DeviceAwareWorkspaceConfiguration mirroredWorkspaceConfig(long initialSize) {
        return DeviceAwareWorkspaceConfiguration.mirrored(initialSize);
    }

    @Override
    public String toString() {
        return String.format("BackendRegistry[backends=%d, devices=%d, cpuDefault=%s, gpuDefault=%s]",
                backends.size(),
                devicesById.size(),
                defaultCpuDevice != null ? defaultCpuDevice.getDeviceId() : "none",
                defaultGpuDevice != null ? defaultGpuDevice.getDeviceId() : "none");
    }
}
