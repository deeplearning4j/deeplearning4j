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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.abstracts.DefaultMultiBackendWorkspace;
import org.nd4j.linalg.api.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.api.memory.conf.DeviceAwareWorkspaceConfiguration;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.factory.BackendRegistry;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manager for multi-backend workspaces.
 * Provides thread-local workspace management with device awareness.
 */
@Slf4j
public class MultiBackendWorkspaceManager {

    private static volatile MultiBackendWorkspaceManager instance;
    private static final Object INSTANCE_LOCK = new Object();

    // Thread-local workspace storage: threadId -> workspaceId -> workspace
    private final Map<Long, Map<String, MultiBackendWorkspace>> threadWorkspaces = new ConcurrentHashMap<>();

    // Global workspace registry for cross-thread access
    private final Map<String, MultiBackendWorkspace> globalWorkspaces = new ConcurrentHashMap<>();

    // Default configurations
    private volatile DeviceAwareWorkspaceConfiguration defaultConfiguration;

    private MultiBackendWorkspaceManager() {
        this.defaultConfiguration = DeviceAwareWorkspaceConfiguration.builder()
                .initialSize(0)
                .maxSize(Long.MAX_VALUE)
                .build();
    }

    /**
     * Get the singleton instance
     * @return the manager instance
     */
    public static MultiBackendWorkspaceManager getInstance() {
        if (instance == null) {
            synchronized (INSTANCE_LOCK) {
                if (instance == null) {
                    instance = new MultiBackendWorkspaceManager();
                }
            }
        }
        return instance;
    }

    /**
     * Create a new multi-backend workspace
     * @param configuration the workspace configuration
     * @return the created workspace
     */
    public MultiBackendWorkspace createWorkspace(@NonNull DeviceAwareWorkspaceConfiguration configuration) {
        return createWorkspace(configuration, UUID.randomUUID().toString());
    }

    /**
     * Create a new multi-backend workspace with ID
     * @param configuration the workspace configuration
     * @param id the workspace ID
     * @return the created workspace
     */
    public MultiBackendWorkspace createWorkspace(@NonNull DeviceAwareWorkspaceConfiguration configuration,
                                                  @NonNull String id) {
        MultiBackendWorkspace workspace = new DefaultMultiBackendWorkspace(configuration, id);

        // Register in thread-local storage
        long threadId = Thread.currentThread().getId();
        threadWorkspaces.computeIfAbsent(threadId, k -> new ConcurrentHashMap<>())
                .put(id, workspace);

        log.debug("Created multi-backend workspace {} on thread {}", id, threadId);
        return workspace;
    }

    /**
     * Create a workspace from standard configuration (auto-converts to device-aware)
     * @param configuration standard configuration
     * @param id the workspace ID
     * @return the created workspace
     */
    public MultiBackendWorkspace createWorkspace(@NonNull WorkspaceConfiguration configuration,
                                                  @NonNull String id) {
        DeviceAwareWorkspaceConfiguration deviceConfig = DeviceAwareWorkspaceConfiguration.fromBase(configuration);
        return createWorkspace(deviceConfig, id);
    }

    /**
     * Get or create a workspace for the current thread
     * @param id the workspace ID
     * @return the workspace
     */
    public MultiBackendWorkspace getWorkspace(@NonNull String id) {
        long threadId = Thread.currentThread().getId();
        Map<String, MultiBackendWorkspace> workspaces = threadWorkspaces.get(threadId);
        return workspaces != null ? workspaces.get(id) : null;
    }

    /**
     * Get or create a workspace with configuration
     * @param configuration the configuration
     * @param id the workspace ID
     * @return the workspace (created if doesn't exist)
     */
    public MultiBackendWorkspace getOrCreateWorkspace(@NonNull DeviceAwareWorkspaceConfiguration configuration,
                                                       @NonNull String id) {
        MultiBackendWorkspace workspace = getWorkspace(id);
        if (workspace == null) {
            workspace = createWorkspace(configuration, id);
        }
        return workspace;
    }

    /**
     * Get and activate a workspace (enter scope)
     * @param configuration the configuration
     * @param id the workspace ID
     * @return the activated workspace
     */
    public MultiBackendWorkspace getAndActivateWorkspace(@NonNull DeviceAwareWorkspaceConfiguration configuration,
                                                          @NonNull String id) {
        MultiBackendWorkspace workspace = getOrCreateWorkspace(configuration, id);
        workspace.notifyScopeEntered();
        return workspace;
    }

    /**
     * Get and activate a workspace with default configuration
     * @param id the workspace ID
     * @return the activated workspace
     */
    public MultiBackendWorkspace getAndActivateWorkspace(@NonNull String id) {
        return getAndActivateWorkspace(defaultConfiguration, id);
    }

    /**
     * Create a workspace on a specific device
     * @param configuration the configuration
     * @param id the workspace ID
     * @param device the target device
     * @return the created workspace
     */
    public MultiBackendWorkspace createWorkspaceOnDevice(@NonNull DeviceAwareWorkspaceConfiguration configuration,
                                                          @NonNull String id,
                                                          @NonNull DeviceDescriptor device) {
        // Create a new configuration with the specified primary device
        DeviceAwareWorkspaceConfiguration deviceConfig = DeviceAwareWorkspaceConfiguration.builder()
                .policyAllocation(configuration.getPolicyAllocation())
                .policySpill(configuration.getPolicySpill())
                .policyMirroring(configuration.getPolicyMirroring())
                .policyLearning(configuration.getPolicyLearning())
                .policyReset(configuration.getPolicyReset())
                .policyLocation(configuration.getPolicyLocation())
                .initialSize(configuration.getInitialSize())
                .minSize(configuration.getMinSize())
                .maxSize(configuration.getMaxSize())
                .overallocationLimit(configuration.getOverallocationLimit())
                .cyclesBeforeInitialization(configuration.getCyclesBeforeInitialization())
                .primaryDevice(device)
                .preferredDeviceTypes(configuration.getPreferredDeviceTypes())
                .transferPolicy(configuration.getTransferPolicy())
                .deviceSelectionPolicy(configuration.getDeviceSelectionPolicy())
                .build();
        return createWorkspace(deviceConfig, id);
    }

    /**
     * Destroy a workspace
     * @param workspace the workspace to destroy
     */
    public void destroyWorkspace(@NonNull MultiBackendWorkspace workspace) {
        String id = workspace.getId();

        // Remove from thread-local storage
        long threadId = Thread.currentThread().getId();
        Map<String, MultiBackendWorkspace> workspaces = threadWorkspaces.get(threadId);
        if (workspaces != null) {
            workspaces.remove(id);
        }

        // Remove from global registry
        globalWorkspaces.remove(id);

        // Destroy the workspace
        workspace.destroyWorkspace();

        log.debug("Destroyed multi-backend workspace {}", id);
    }

    /**
     * Destroy all workspaces for the current thread
     */
    public void destroyAllWorkspacesForCurrentThread() {
        long threadId = Thread.currentThread().getId();
        Map<String, MultiBackendWorkspace> workspaces = threadWorkspaces.remove(threadId);
        if (workspaces != null) {
            for (MultiBackendWorkspace workspace : workspaces.values()) {
                workspace.destroyWorkspace();
            }
        }
    }

    /**
     * Check if a workspace exists
     * @param id the workspace ID
     * @return true if exists
     */
    public boolean workspaceExists(@NonNull String id) {
        return getWorkspace(id) != null;
    }

    /**
     * Get all workspace IDs for the current thread
     * @return set of workspace IDs
     */
    public Set<String> getWorkspaceIdsForCurrentThread() {
        long threadId = Thread.currentThread().getId();
        Map<String, MultiBackendWorkspace> workspaces = threadWorkspaces.get(threadId);
        return workspaces != null ? new HashSet<>(workspaces.keySet()) : Collections.emptySet();
    }

    /**
     * Make a workspace global (accessible from other threads)
     * @param workspace the workspace
     */
    public void makeGlobal(@NonNull MultiBackendWorkspace workspace) {
        globalWorkspaces.put(workspace.getId(), workspace);
    }

    /**
     * Get a global workspace
     * @param id the workspace ID
     * @return the workspace, or null if not found
     */
    public MultiBackendWorkspace getGlobalWorkspace(@NonNull String id) {
        return globalWorkspaces.get(id);
    }

    /**
     * Set the default configuration for new workspaces
     * @param configuration the default configuration
     */
    public void setDefaultConfiguration(@NonNull DeviceAwareWorkspaceConfiguration configuration) {
        this.defaultConfiguration = configuration;
    }

    /**
     * Get the default configuration
     * @return default configuration
     */
    public DeviceAwareWorkspaceConfiguration getDefaultConfiguration() {
        return defaultConfiguration;
    }

    /**
     * Print allocation statistics for all workspaces on current thread
     */
    public void printAllocationStatistics() {
        long threadId = Thread.currentThread().getId();
        Map<String, MultiBackendWorkspace> workspaces = threadWorkspaces.get(threadId);

        if (workspaces == null || workspaces.isEmpty()) {
            log.info("No multi-backend workspaces for thread {}", threadId);
            return;
        }

        log.info("Multi-Backend Workspace Statistics for thread {}:", threadId);
        for (Map.Entry<String, MultiBackendWorkspace> entry : workspaces.entrySet()) {
            Map<String, Object> stats = entry.getValue().getStatistics();
            log.info("  Workspace {}: {}", entry.getKey(), stats);
        }
    }

    /**
     * Get total memory usage across all workspaces for current thread
     * @return total memory in bytes
     */
    public long getTotalMemoryUsage() {
        long total = 0;
        long threadId = Thread.currentThread().getId();
        Map<String, MultiBackendWorkspace> workspaces = threadWorkspaces.get(threadId);

        if (workspaces != null) {
            for (MultiBackendWorkspace workspace : workspaces.values()) {
                total += workspace.getCurrentSize();
            }
        }
        return total;
    }

    /**
     * Get memory usage per device across all workspaces
     * @return map of device ID to total memory
     */
    public Map<String, Long> getMemoryUsagePerDevice() {
        Map<String, Long> usage = new HashMap<>();
        long threadId = Thread.currentThread().getId();
        Map<String, MultiBackendWorkspace> workspaces = threadWorkspaces.get(threadId);

        if (workspaces != null) {
            for (MultiBackendWorkspace workspace : workspaces.values()) {
                for (DeviceDescriptor device : workspace.getActiveDevices()) {
                    String deviceId = device.getDeviceId();
                    long deviceUsage = workspace.getAllocatedSizeForDevice(device);
                    usage.merge(deviceId, deviceUsage, Long::sum);
                }
            }
        }
        return usage;
    }

    /**
     * Handle memory pressure callbacks by releasing or compacting workspaces on a device.
     *
     * @param device the device under pressure
     * @param utilization current utilization (0.0 to 1.0)
     */
    public void handleMemoryPressure(@NonNull DeviceDescriptor device, double utilization) {
        log.debug("Handling memory pressure on device {} (utilization {})", device.getDeviceId(), utilization);

        Set<MultiBackendWorkspace> uniqueWorkspaces =
                Collections.newSetFromMap(new IdentityHashMap<>());
        for (Map<String, MultiBackendWorkspace> workspaces : threadWorkspaces.values()) {
            if (workspaces != null) {
                uniqueWorkspaces.addAll(workspaces.values());
            }
        }
        uniqueWorkspaces.addAll(globalWorkspaces.values());

        for (MultiBackendWorkspace workspace : uniqueWorkspaces) {
            if (workspace == null || !workspace.hasMemoryOnDevice(device)) {
                continue;
            }

            try {
                DeviceDescriptor primary = workspace.getPrimaryDevice();
                boolean isPrimary = primary != null && device.getDeviceId().equals(primary.getDeviceId());
                if (isPrimary) {
                    workspace.compactOnDevice(device);
                } else {
                    workspace.releaseOnDevice(device);
                    DeviceAwareWorkspaceConfiguration config = workspace.getMultiBackendConfiguration();
                    if (workspace.isMirroringEnabled()
                            && config.getMirrorDevices() != null) {
                        for (DeviceDescriptor mirrorDevice : config.getMirrorDevices()) {
                            if (device.getDeviceId().equals(mirrorDevice.getDeviceId())) {
                                workspace.setMirroringEnabled(false);
                                break;
                            }
                        }
                    }
                }
            } catch (Exception e) {
                log.warn("Failed to handle memory pressure for workspace {} on device {}",
                        workspace.getId(), device.getDeviceId(), e);
            }
        }
    }

    /**
     * Scope out of all workspaces (leave all active scopes)
     * @return a dummy workspace representing "outside all workspaces"
     */
    public MemoryWorkspace scopeOutOfWorkspaces() {
        long threadId = Thread.currentThread().getId();
        Map<String, MultiBackendWorkspace> workspaces = threadWorkspaces.get(threadId);

        if (workspaces != null) {
            for (MultiBackendWorkspace workspace : workspaces.values()) {
                if (workspace.isScopeActive()) {
                    workspace.notifyScopeLeft();
                }
            }
        }

        // Return a dummy workspace
        return new DummyWorkspace();
    }
}
