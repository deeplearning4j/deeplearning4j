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
// @author Adam Gibson
//
// Device Manager - Centralized device discovery and management for model parallelism
//

#ifndef SD_DEVICE_MANAGER_H
#define SD_DEVICE_MANAGER_H

#include <system/common.h>
#include <execution/Engine.h>
#include <execution/ModelParallelConfig.h>

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <functional>

namespace sd {
namespace modelparallel {

/**
 * Detailed device information
 */
struct SD_LIB_EXPORT DeviceInfo {
    // Basic identification
    DeviceType type = DeviceType::CPU;
    int deviceIndex = 0;
    int globalIndex = 0;          // Global index across all device types
    std::string name;
    std::string vendor;
    std::string driverVersion;

    // Memory information
    size_t totalMemory = 0;
    size_t freeMemory = 0;
    size_t availableMemory = 0;   // Memory available for allocation

    // Compute capabilities
    int computeCapabilityMajor = 0;
    int computeCapabilityMinor = 0;
    int numCores = 0;
    int numSMs = 0;               // For CUDA GPUs
    float clockSpeedGHz = 0.0f;
    float theoreticalTFlops = 0.0f;

    // P2P and interconnect info
    bool supportsP2P = false;
    bool supportsUnifiedMemory = false;
    bool supportsAsyncCopy = false;
    std::vector<int> p2pConnectedDevices;  // Devices this can do P2P with

    // Availability status
    bool available = true;
    bool inUse = false;
    int currentUserCount = 0;

    // Backend engine mapping
    samediff::Engine engine = samediff::ENGINE_CPU;

    bool operator==(const DeviceInfo& other) const {
        return type == other.type && deviceIndex == other.deviceIndex;
    }
};

/**
 * P2P connection information between two devices
 */
struct SD_LIB_EXPORT P2PConnection {
    int sourceDevice = -1;
    int targetDevice = -1;
    bool directAccess = false;    // Can access memory directly
    bool nvlinkConnected = false; // Connected via NVLink
    float bandwidth = 0.0f;       // Estimated bandwidth in GB/s
    float latency = 0.0f;         // Estimated latency in microseconds
    bool bidirectional = true;
};

/**
 * Device group for collective operations
 */
struct SD_LIB_EXPORT DeviceGroup {
    std::string name;
    std::vector<int> deviceIndices;  // Global device indices
    CommunicationPattern defaultPattern = CommunicationPattern::ALL_REDUCE;
    bool initialized = false;

    int size() const { return static_cast<int>(deviceIndices.size()); }
    bool contains(int deviceIndex) const {
        for (int idx : deviceIndices) {
            if (idx == deviceIndex) return true;
        }
        return false;
    }
};

/**
 * Device allocation result
 */
struct SD_LIB_EXPORT DeviceAllocation {
    bool success = false;
    std::vector<DeviceInfo> allocatedDevices;
    std::string errorMessage;

    int size() const { return static_cast<int>(allocatedDevices.size()); }
    bool empty() const { return allocatedDevices.empty(); }
};

/**
 * Device Manager - Singleton for managing all compute devices
 *
 * This class provides:
 * - Device discovery and enumeration
 * - Device capability querying
 * - P2P connection management
 * - Device allocation for parallel execution
 * - Memory tracking across devices
 */
class SD_LIB_EXPORT DeviceManager {
public:
    /**
     * Get the singleton instance
     */
    static DeviceManager& getInstance();

    /**
     * Initialize the device manager (discovers all available devices)
     */
    bool initialize();

    /**
     * Refresh device information (re-enumerate and update status)
     */
    void refresh();

    /**
     * Shutdown and release resources
     */
    void shutdown();

    /**
     * Check if manager is initialized
     */
    bool isInitialized() const { return _initialized.load(); }

    // =========================================================================
    // Device Enumeration
    // =========================================================================

    /**
     * Get total number of devices (all types)
     */
    int getTotalDeviceCount() const;

    /**
     * Get number of devices of a specific type
     */
    int getDeviceCount(DeviceType type) const;

    /**
     * Get number of CPU devices (logical processors available)
     */
    int getCpuCount() const;

    /**
     * Get number of CUDA GPUs
     */
    int getCudaGpuCount() const;

    /**
     * Get all devices
     */
    std::vector<DeviceInfo> getAllDevices() const;

    /**
     * Get devices of a specific type
     */
    std::vector<DeviceInfo> getDevices(DeviceType type) const;

    /**
     * Get device info by global index
     */
    DeviceInfo getDeviceInfo(int globalIndex) const;

    /**
     * Get device info by type and local index
     */
    DeviceInfo getDeviceInfo(DeviceType type, int localIndex) const;

    /**
     * Check if a device exists and is available
     */
    bool isDeviceAvailable(DeviceType type, int localIndex) const;

    /**
     * Check if a device is currently in use
     */
    bool isDeviceInUse(int globalIndex) const;

    // =========================================================================
    // Device Capabilities
    // =========================================================================

    /**
     * Get parallel capabilities of the system
     */
    ParallelCapabilities getParallelCapabilities() const;

    /**
     * Check if tensor parallelism is supported
     */
    bool supportsTensorParallel() const;

    /**
     * Check if pipeline parallelism is supported
     */
    bool supportsPipelineParallel() const;

    /**
     * Check if P2P is supported between two devices
     */
    bool supportsP2P(int device1, int device2) const;

    /**
     * Get P2P connection info between two devices
     */
    P2PConnection getP2PConnection(int device1, int device2) const;

    /**
     * Get all P2P connections
     */
    std::vector<P2PConnection> getAllP2PConnections() const;

    /**
     * Enable P2P between two devices
     */
    bool enableP2P(int device1, int device2);

    /**
     * Disable P2P between two devices
     */
    void disableP2P(int device1, int device2);

    /**
     * Enable P2P for all compatible device pairs
     */
    void enableAllP2P();

    // =========================================================================
    // Device Allocation
    // =========================================================================

    /**
     * Allocate devices for a parallel configuration
     */
    DeviceAllocation allocateDevices(const ModelParallelConfig& config);

    /**
     * Allocate specific number of devices of a type
     */
    DeviceAllocation allocateDevices(DeviceType type, int count);

    /**
     * Allocate a device group for collective operations
     */
    DeviceGroup createDeviceGroup(const std::string& name, const std::vector<DeviceSpec>& specs);

    /**
     * Release a device group
     */
    void releaseDeviceGroup(const std::string& name);

    /**
     * Get a device group by name
     */
    DeviceGroup getDeviceGroup(const std::string& name) const;

    /**
     * Mark device as in use
     */
    void acquireDevice(int globalIndex);

    /**
     * Mark device as no longer in use
     */
    void releaseDevice(int globalIndex);

    /**
     * Release all acquired devices
     */
    void releaseAllDevices();

    // =========================================================================
    // Memory Management
    // =========================================================================

    /**
     * Get total memory across all devices of a type
     */
    size_t getTotalMemory(DeviceType type) const;

    /**
     * Get free memory across all devices of a type
     */
    size_t getFreeMemory(DeviceType type) const;

    /**
     * Get free memory on a specific device
     */
    size_t getDeviceFreeMemory(int globalIndex) const;

    /**
     * Update memory stats for a device
     */
    void updateMemoryStats(int globalIndex);

    /**
     * Update memory stats for all devices
     */
    void updateAllMemoryStats();

    /**
     * Reserve memory on a device
     */
    bool reserveMemory(int globalIndex, size_t bytes);

    /**
     * Release reserved memory on a device
     */
    void releaseReservedMemory(int globalIndex, size_t bytes);

    // =========================================================================
    // Device Selection
    // =========================================================================

    /**
     * Find best devices for a workload
     * @param requiredMemory Memory required per device
     * @param count Number of devices needed
     * @param preferP2P Prefer devices with P2P support
     */
    std::vector<DeviceInfo> findBestDevices(size_t requiredMemory, int count, bool preferP2P = true);

    /**
     * Get the best available GPU
     */
    DeviceInfo getBestGpu() const;

    /**
     * Set current device for this thread
     */
    void setCurrentDevice(int globalIndex);

    /**
     * Get current device for this thread
     */
    int getCurrentDevice() const;

    /**
     * Execute a function on a specific device
     */
    template<typename Func>
    auto executeOnDevice(int globalIndex, Func&& func) -> decltype(func()) {
        int previousDevice = getCurrentDevice();
        setCurrentDevice(globalIndex);
        auto result = func();
        setCurrentDevice(previousDevice);
        return result;
    }

    // =========================================================================
    // Engine Mapping
    // =========================================================================

    /**
     * Get the Engine enum for a device type
     */
    static samediff::Engine deviceTypeToEngine(DeviceType type);

    /**
     * Get device type from Engine enum
     */
    static DeviceType engineToDeviceType(samediff::Engine engine);

    /**
     * Get global device index from Engine and local index
     */
    int getGlobalIndex(samediff::Engine engine, int localIndex) const;

    // =========================================================================
    // Utility
    // =========================================================================

    /**
     * Synchronize all devices
     */
    void synchronizeAll();

    /**
     * Synchronize a specific device
     */
    void synchronize(int globalIndex);

    /**
     * Get string representation of device info
     */
    std::string deviceToString(const DeviceInfo& device) const;

    /**
     * Print all device information
     */
    void printDeviceInfo() const;

private:
    DeviceManager();
    ~DeviceManager();

    // Prevent copying
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    // Device discovery methods
    void discoverCpuDevices();
    void discoverCudaDevices();
    void discoverMetalDevices();
    void discoverVulkanDevices();
    void discoverOpenClDevices();

    // P2P initialization
    void initializeP2PConnections();
    void probeP2PCapabilities(int device1, int device2);

    // CUDA dispatch helper (implemented in cuda/DeviceManager_cuda.cu)
    void setCurrentDeviceCuda(const DeviceInfo& device);

    // Internal device lookup
    int findGlobalIndex(DeviceType type, int localIndex) const;
    DeviceInfo* findDevice(int globalIndex);
    const DeviceInfo* findDevice(int globalIndex) const;

    // Member variables
    std::atomic<bool> _initialized{false};
    mutable std::mutex _mutex;

    std::vector<DeviceInfo> _devices;
    std::unordered_map<DeviceType, std::vector<int>> _devicesByType;  // Type -> global indices
    std::vector<P2PConnection> _p2pConnections;
    std::unordered_map<std::string, DeviceGroup> _deviceGroups;

    // Thread-local current device (accessor pattern for MSVC C2492 compatibility)
    static int& currentDeviceRef();

    // Reserved memory tracking
    std::unordered_map<int, size_t> _reservedMemory;
};

/**
 * RAII guard for device context
 */
class SD_LIB_EXPORT DeviceContextGuard {
public:
    explicit DeviceContextGuard(int globalIndex);
    DeviceContextGuard(DeviceType type, int localIndex);
    ~DeviceContextGuard();

    // Prevent copying
    DeviceContextGuard(const DeviceContextGuard&) = delete;
    DeviceContextGuard& operator=(const DeviceContextGuard&) = delete;

    int getDevice() const { return _deviceIndex; }

private:
    int _deviceIndex;
    int _previousDevice;
};

/**
 * Helper function to get the global device manager
 */
inline DeviceManager& getDeviceManager() {
    return DeviceManager::getInstance();
}

/**
 * Helper to check if multi-GPU is available
 */
inline bool hasMultiGpu() {
    return DeviceManager::getInstance().getCudaGpuCount() > 1;
}

/**
 * Helper to check if model parallelism is feasible
 */
inline bool canDoModelParallel() {
    auto& dm = DeviceManager::getInstance();
    return dm.getCudaGpuCount() > 1 || dm.getCpuCount() > 1;
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_DEVICE_MANAGER_H
