/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Global manager for multi-backend workspaces across threads and devices
//

#ifndef LIBND4J_DEVICE_WORKSPACE_MANAGER_H
#define LIBND4J_DEVICE_WORKSPACE_MANAGER_H

#include <memory/MultiBackendWorkspace.h>
#include <system/common.h>

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <thread>
#include <string>
#include <vector>
#include <memory>

namespace sd {
namespace memory {

/**
 * Thread-local workspace registry entry
 */
struct ThreadWorkspaceEntry {
    std::unordered_map<std::string, MultiBackendWorkspace*> workspaces;
    std::mutex mutex;
};

/**
 * Global manager for multi-backend workspaces.
 * Provides thread-local workspace management with device awareness.
 * Singleton pattern with proper cleanup.
 */
class SD_LIB_EXPORT DeviceWorkspaceManager {
private:
    // Singleton instance
    static DeviceWorkspaceManager* _instance;
    static std::mutex _instanceMutex;

    // Thread-local workspace storage: threadId -> workspaces
    std::unordered_map<std::thread::id, std::unique_ptr<ThreadWorkspaceEntry>> _threadWorkspaces;
    mutable std::shared_mutex _threadWorkspacesMutex;

    // Global workspace registry (cross-thread accessible)
    std::unordered_map<std::string, MultiBackendWorkspace*> _globalWorkspaces;
    mutable std::mutex _globalWorkspacesMutex;

    // Default configuration
    MultiBackendWorkspaceConfig _defaultConfig;
    std::mutex _configMutex;

    // Statistics
    std::atomic<sd::LongType> _totalWorkspacesCreated;
    std::atomic<sd::LongType> _totalWorkspacesDestroyed;
    std::atomic<sd::LongType> _totalMemoryAllocated;

    // Private constructor for singleton
    DeviceWorkspaceManager();

    // Get or create thread-local entry
    ThreadWorkspaceEntry* getThreadEntry();

public:
    // Prevent copying
    DeviceWorkspaceManager(const DeviceWorkspaceManager&) = delete;
    DeviceWorkspaceManager& operator=(const DeviceWorkspaceManager&) = delete;

    /**
     * Destructor - cleans up all workspaces
     */
    ~DeviceWorkspaceManager();

    /**
     * Get the singleton instance
     */
    static DeviceWorkspaceManager* getInstance();

    /**
     * Shutdown and cleanup the singleton (call at program termination)
     */
    static void shutdown();

    // ========================
    // Workspace Creation
    // ========================

    /**
     * Create a new multi-backend workspace with configuration
     */
    MultiBackendWorkspace* createWorkspace(const MultiBackendWorkspaceConfig& config,
                                            const std::string& id = "");

    /**
     * Create a workspace with simple parameters
     */
    MultiBackendWorkspace* createWorkspace(sd::LongType initialSize,
                                            const std::string& id = "");

    /**
     * Create a workspace on a specific device
     */
    MultiBackendWorkspace* createWorkspaceOnDevice(const MultiBackendWorkspaceConfig& config,
                                                    const std::string& id,
                                                    const DeviceDescriptor& device);

    // ========================
    // Workspace Retrieval
    // ========================

    /**
     * Get a workspace by ID for the current thread
     */
    MultiBackendWorkspace* getWorkspace(const std::string& id);

    /**
     * Get or create a workspace
     */
    MultiBackendWorkspace* getOrCreateWorkspace(const MultiBackendWorkspaceConfig& config,
                                                 const std::string& id);

    /**
     * Get and activate a workspace (enter scope)
     */
    MultiBackendWorkspace* getAndActivateWorkspace(const MultiBackendWorkspaceConfig& config,
                                                    const std::string& id);

    /**
     * Get and activate with default configuration
     */
    MultiBackendWorkspace* getAndActivateWorkspace(const std::string& id);

    /**
     * Check if a workspace exists
     */
    bool workspaceExists(const std::string& id);

    // ========================
    // Workspace Destruction
    // ========================

    /**
     * Destroy a workspace
     */
    void destroyWorkspace(MultiBackendWorkspace* workspace);

    /**
     * Destroy a workspace by ID
     */
    void destroyWorkspace(const std::string& id);

    /**
     * Destroy all workspaces for the current thread
     */
    void destroyAllWorkspacesForCurrentThread();

    /**
     * Destroy all workspaces (global cleanup)
     */
    void destroyAllWorkspaces();

    // ========================
    // Global Workspace Registry
    // ========================

    /**
     * Make a workspace globally accessible (cross-thread)
     */
    void makeGlobal(MultiBackendWorkspace* workspace);

    /**
     * Get a global workspace
     */
    MultiBackendWorkspace* getGlobalWorkspace(const std::string& id);

    /**
     * Remove workspace from global registry
     */
    void removeFromGlobal(const std::string& id);

    // ========================
    // Configuration
    // ========================

    /**
     * Set the default configuration
     */
    void setDefaultConfiguration(const MultiBackendWorkspaceConfig& config);

    /**
     * Get the default configuration
     */
    MultiBackendWorkspaceConfig getDefaultConfiguration();

    // ========================
    // Statistics and Info
    // ========================

    /**
     * Get all workspace IDs for the current thread
     */
    std::vector<std::string> getWorkspaceIdsForCurrentThread();

    /**
     * Get total memory usage across all workspaces
     */
    sd::LongType getTotalMemoryUsage();

    /**
     * Get memory usage per device
     */
    std::unordered_map<std::string, sd::LongType> getMemoryUsagePerDevice();

    /**
     * Get total workspaces created
     */
    sd::LongType getTotalWorkspacesCreated() const { return _totalWorkspacesCreated.load(); }

    /**
     * Get total workspaces destroyed
     */
    sd::LongType getTotalWorkspacesDestroyed() const { return _totalWorkspacesDestroyed.load(); }

    /**
     * Print allocation statistics
     */
    void printAllocationStatistics();

    /**
     * Scope out of all workspaces (leave all active scopes)
     */
    void scopeOutOfAllWorkspaces();
};

/**
 * Opaque handle for device workspace manager
 */
typedef DeviceWorkspaceManager* DeviceWorkspaceManagerHandle;

// ========================
// C-style API for opaque bindings
// ========================

extern "C" {

/**
 * Get the singleton device workspace manager instance
 */
SD_LIB_EXPORT DeviceWorkspaceManagerHandle getDeviceWorkspaceManager();

/**
 * Shutdown and cleanup the device workspace manager
 */
SD_LIB_EXPORT void shutdownDeviceWorkspaceManager();

/**
 * Create a workspace through the manager
 */
SD_LIB_EXPORT MultiBackendWorkspaceHandle dwmCreateWorkspace(
    DeviceWorkspaceManagerHandle manager,
    sd::LongType initialSize,
    const char* id);

/**
 * Create a workspace with full configuration
 */
SD_LIB_EXPORT MultiBackendWorkspaceHandle dwmCreateWorkspaceWithConfig(
    DeviceWorkspaceManagerHandle manager,
    sd::LongType initialSize,
    sd::LongType maxSize,
    bool crossDeviceMirroring,
    bool asyncTransfers,
    int primaryDeviceType,
    int primaryDeviceIndex,
    const char* id);

/**
 * Get a workspace by ID
 */
SD_LIB_EXPORT MultiBackendWorkspaceHandle dwmGetWorkspace(
    DeviceWorkspaceManagerHandle manager,
    const char* id);

/**
 * Get or create a workspace
 */
SD_LIB_EXPORT MultiBackendWorkspaceHandle dwmGetOrCreateWorkspace(
    DeviceWorkspaceManagerHandle manager,
    sd::LongType initialSize,
    const char* id);

/**
 * Get and activate a workspace
 */
SD_LIB_EXPORT MultiBackendWorkspaceHandle dwmGetAndActivateWorkspace(
    DeviceWorkspaceManagerHandle manager,
    sd::LongType initialSize,
    const char* id);

/**
 * Check if workspace exists
 */
SD_LIB_EXPORT bool dwmWorkspaceExists(
    DeviceWorkspaceManagerHandle manager,
    const char* id);

/**
 * Destroy a workspace
 */
SD_LIB_EXPORT void dwmDestroyWorkspace(
    DeviceWorkspaceManagerHandle manager,
    MultiBackendWorkspaceHandle workspace);

/**
 * Destroy a workspace by ID
 */
SD_LIB_EXPORT void dwmDestroyWorkspaceById(
    DeviceWorkspaceManagerHandle manager,
    const char* id);

/**
 * Destroy all workspaces for current thread
 */
SD_LIB_EXPORT void dwmDestroyAllWorkspacesForCurrentThread(
    DeviceWorkspaceManagerHandle manager);

/**
 * Make workspace global
 */
SD_LIB_EXPORT void dwmMakeGlobal(
    DeviceWorkspaceManagerHandle manager,
    MultiBackendWorkspaceHandle workspace);

/**
 * Get global workspace
 */
SD_LIB_EXPORT MultiBackendWorkspaceHandle dwmGetGlobalWorkspace(
    DeviceWorkspaceManagerHandle manager,
    const char* id);

/**
 * Get total memory usage
 */
SD_LIB_EXPORT sd::LongType dwmGetTotalMemoryUsage(DeviceWorkspaceManagerHandle manager);

/**
 * Get workspace count for current thread
 */
SD_LIB_EXPORT int dwmGetWorkspaceCountForCurrentThread(DeviceWorkspaceManagerHandle manager);

/**
 * Scope out of all workspaces
 */
SD_LIB_EXPORT void dwmScopeOutOfAllWorkspaces(DeviceWorkspaceManagerHandle manager);

}  // extern "C"

}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_DEVICE_WORKSPACE_MANAGER_H
