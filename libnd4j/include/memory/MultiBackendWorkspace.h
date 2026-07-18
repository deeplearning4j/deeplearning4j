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
// Multi-backend workspace for device-aware memory management
//

#ifndef LIBND4J_MULTI_BACKEND_WORKSPACE_H
#define LIBND4J_MULTI_BACKEND_WORKSPACE_H

#include <memory/Workspace.h>
#include <memory/MemoryType.h>
#include <system/common.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace sd {
namespace memory {

/**
 * Coherence state for multi-device memory (MSI-like protocol)
 */
enum class CoherenceState : int {
    INVALID = 0,    // Data is stale/not present
    SHARED = 1,     // Data is valid for reading, shared with other devices
    EXCLUSIVE = 2,  // Data is valid, only copy
    MODIFIED = 3    // Data has been modified, only valid copy
};

/**
 * Device type enumeration
 */
enum class DeviceType : int {
    CPU = 0,
    CUDA_GPU = 1,
    ROCM_GPU = 2,
    TPU = 3,
    VULKAN_GPU = 4,
    UNKNOWN = 99
};

/**
 * Device descriptor - identifies a specific compute device
 */
struct SD_LIB_EXPORT DeviceDescriptor {
    DeviceType deviceType;
    int deviceIndex;
    std::string deviceId;

    DeviceDescriptor() : deviceType(DeviceType::CPU), deviceIndex(0), deviceId("cpu:0") {}
    DeviceDescriptor(DeviceType type, int index)
        : deviceType(type), deviceIndex(index) {
        const char* prefix = type == DeviceType::CPU ? "cpu:" :
                             type == DeviceType::CUDA_GPU ? "cuda:" :
                             type == DeviceType::VULKAN_GPU ? "vulkan:" : "device:";
        deviceId = std::string(prefix) + std::to_string(index);
    }

    bool operator==(const DeviceDescriptor& other) const {
        return deviceType == other.deviceType && deviceIndex == other.deviceIndex;
    }

    bool operator!=(const DeviceDescriptor& other) const {
        return !(*this == other);
    }
};

/**
 * Hash function for DeviceDescriptor (for use in unordered_map)
 */
struct DeviceDescriptorHash {
    std::size_t operator()(const DeviceDescriptor& d) const {
        return std::hash<int>()(static_cast<int>(d.deviceType)) ^
               (std::hash<int>()(d.deviceIndex) << 1);
    }
};

/**
 * Per-device workspace allocation info
 */
struct DeviceAllocation {
    Workspace* workspace;           // The underlying workspace
    CoherenceState coherenceState;  // Current coherence state
    sd::LongType version;           // Version number for coherence
    bool isOwned;                   // Whether we own the workspace (for cleanup)

    DeviceAllocation()
        : workspace(nullptr), coherenceState(CoherenceState::INVALID),
          version(0), isOwned(false) {}
};

/**
 * Configuration for multi-backend workspace
 */
struct SD_LIB_EXPORT MultiBackendWorkspaceConfig {
    sd::LongType initialSize;
    sd::LongType maxSize;
    bool crossDeviceMirroring;
    bool asyncTransfers;
    DeviceDescriptor primaryDevice;

    MultiBackendWorkspaceConfig()
        : initialSize(0), maxSize(LLONG_MAX),
          crossDeviceMirroring(false), asyncTransfers(true) {}
};

/**
 * Multi-backend workspace that manages memory across multiple devices.
 * Provides coherence tracking and cross-device transfers.
 */
class SD_LIB_EXPORT MultiBackendWorkspace {
protected:
    std::string _id;
    MultiBackendWorkspaceConfig _config;

    // Per-device workspaces
    std::unordered_map<DeviceDescriptor, DeviceAllocation, DeviceDescriptorHash> _deviceAllocations;

    // Primary device (owner of authoritative data)
    DeviceDescriptor _primaryDevice;

    // Global version counter for coherence
    std::atomic<sd::LongType> _globalVersion;

    // Scope tracking
    std::atomic<int> _scopeDepth;
    std::atomic<bool> _scopeActive;

    // Thread safety
    mutable std::mutex _mutex;
    mutable std::mutex _transferMutex;

    // Statistics
    std::atomic<sd::LongType> _totalAllocations;
    std::atomic<sd::LongType> _totalDeallocations;
    std::atomic<sd::LongType> _totalTransfers;

    // Internal helpers
    void initDeviceWorkspace(const DeviceDescriptor& device, sd::LongType size);
    void freeDeviceWorkspace(const DeviceDescriptor& device);
    void invalidateOtherDevices(const DeviceDescriptor& exceptDevice);

public:
    /**
     * Constructor with configuration
     */
    explicit MultiBackendWorkspace(const MultiBackendWorkspaceConfig& config,
                                    const std::string& id = "");

    /**
     * Constructor with simple parameters
     */
    MultiBackendWorkspace(sd::LongType initialSize = 0,
                          const DeviceDescriptor& primaryDevice = DeviceDescriptor());

    /**
     * Destructor - cleans up all device allocations
     */
    ~MultiBackendWorkspace();

    // Prevent copying
    MultiBackendWorkspace(const MultiBackendWorkspace&) = delete;
    MultiBackendWorkspace& operator=(const MultiBackendWorkspace&) = delete;

    // Allow moving
    MultiBackendWorkspace(MultiBackendWorkspace&& other) noexcept;
    MultiBackendWorkspace& operator=(MultiBackendWorkspace&& other) noexcept;

    // ========================
    // Identification
    // ========================

    const std::string& getId() const { return _id; }
    void setId(const std::string& id) { _id = id; }

    // ========================
    // Device Management
    // ========================

    /**
     * Get the primary device
     */
    const DeviceDescriptor& getPrimaryDevice() const { return _primaryDevice; }

    /**
     * Set the primary device
     */
    void setPrimaryDevice(const DeviceDescriptor& device);

    /**
     * Get all devices with allocations
     */
    std::vector<DeviceDescriptor> getActiveDevices() const;

    /**
     * Check if device has an allocation
     */
    bool hasDeviceAllocation(const DeviceDescriptor& device) const;

    // ========================
    // Allocation
    // ========================

    /**
     * Allocate bytes on the primary device
     */
    void* allocateBytes(sd::LongType numBytes);

    /**
     * Allocate bytes on a specific device
     */
    void* allocateBytes(const DeviceDescriptor& device, sd::LongType numBytes);

    /**
     * Allocate bytes with memory type specification
     */
    void* allocateBytes(const DeviceDescriptor& device, MemoryType type, sd::LongType numBytes);

    /**
     * Get the underlying workspace for a device
     */
    Workspace* getDeviceWorkspace(const DeviceDescriptor& device);

    /**
     * Ensure workspace exists on device with minimum size
     */
    void ensureWorkspaceOnDevice(const DeviceDescriptor& device, sd::LongType minSize = 0);

    // ========================
    // Scope Management
    // ========================

    /**
     * Enter workspace scope
     */
    void scopeIn();

    /**
     * Exit workspace scope
     */
    void scopeOut();

    /**
     * Check if scope is active
     */
    bool isScopeActive() const { return _scopeActive.load(); }

    /**
     * Get current scope depth
     */
    int getScopeDepth() const { return _scopeDepth.load(); }

    // ========================
    // Coherence Management
    // ========================

    /**
     * Get coherence state for a device
     */
    CoherenceState getCoherenceState(const DeviceDescriptor& device) const;

    /**
     * Set coherence state for a device
     */
    void setCoherenceState(const DeviceDescriptor& device, CoherenceState state);

    /**
     * Mark data as modified on a device (invalidates others)
     */
    void markModified(const DeviceDescriptor& device);

    /**
     * Invalidate data on a device
     */
    void invalidateDevice(const DeviceDescriptor& device);

    /**
     * Invalidate all devices except the specified one
     */
    void invalidateAllExcept(const DeviceDescriptor& device);

    /**
     * Get the current global version
     */
    sd::LongType getGlobalVersion() const { return _globalVersion.load(); }

    /**
     * Get the version on a specific device
     */
    sd::LongType getDeviceVersion(const DeviceDescriptor& device) const;

    // ========================
    // Transfers
    // ========================

    /**
     * Transfer data from source to target device (synchronous)
     */
    void transferTo(const DeviceDescriptor& source, const DeviceDescriptor& target);

    /**
     * Ensure data is valid on target device (transfer if needed)
     */
    void ensureValidOn(const DeviceDescriptor& device);

    /**
     * Synchronize a device (wait for pending operations)
     */
    void syncDevice(const DeviceDescriptor& device);

    /**
     * Synchronize all devices
     */
    void syncAllDevices();

    // ========================
    // Size and Statistics
    // ========================

    /**
     * Get total allocated size across all devices
     */
    sd::LongType getTotalAllocatedSize() const;

    /**
     * Get allocated size on a specific device
     */
    sd::LongType getAllocatedSizeOnDevice(const DeviceDescriptor& device) const;

    /**
     * Get current offset on primary device
     */
    sd::LongType getCurrentOffset() const;

    /**
     * Get current offset on a specific device
     */
    sd::LongType getCurrentOffsetOnDevice(const DeviceDescriptor& device) const;

    /**
     * Get spilled size on primary device
     */
    sd::LongType getSpilledSize() const;

    /**
     * Get total allocation count
     */
    sd::LongType getTotalAllocationCount() const { return _totalAllocations.load(); }

    /**
     * Get total transfer count
     */
    sd::LongType getTotalTransferCount() const { return _totalTransfers.load(); }

    // ========================
    // Cleanup
    // ========================

    /**
     * Destroy the workspace and free all memory
     */
    void destroy();

    /**
     * Release memory on a specific device
     */
    void releaseOnDevice(const DeviceDescriptor& device);

    /**
     * Reset statistics
     */
    void resetStatistics();

    /**
     * Clone the workspace configuration (not the data)
     */
    MultiBackendWorkspace* clone() const;
};

/**
 * Opaque handle for external bindings (Java/Python)
 */
typedef MultiBackendWorkspace* MultiBackendWorkspaceHandle;

// ========================
// C-style API for opaque bindings
// ========================

extern "C" {

/**
 * Create a new multi-backend workspace
 */
SD_LIB_EXPORT MultiBackendWorkspaceHandle createMultiBackendWorkspace(
    sd::LongType initialSize,
    int primaryDeviceType,
    int primaryDeviceIndex);

/**
 * Create workspace with full configuration
 */
SD_LIB_EXPORT MultiBackendWorkspaceHandle createMultiBackendWorkspaceWithConfig(
    sd::LongType initialSize,
    sd::LongType maxSize,
    bool crossDeviceMirroring,
    bool asyncTransfers,
    int primaryDeviceType,
    int primaryDeviceIndex,
    const char* id);

/**
 * Destroy a multi-backend workspace and free all memory
 */
SD_LIB_EXPORT void destroyMultiBackendWorkspace(MultiBackendWorkspaceHandle handle);

/**
 * Allocate bytes on the primary device
 */
SD_LIB_EXPORT void* mbwAllocateBytes(MultiBackendWorkspaceHandle handle, sd::LongType numBytes);

/**
 * Allocate bytes on a specific device
 */
SD_LIB_EXPORT void* mbwAllocateBytesOnDevice(
    MultiBackendWorkspaceHandle handle,
    sd::LongType numBytes,
    int deviceType,
    int deviceIndex);

/**
 * Enter workspace scope
 */
SD_LIB_EXPORT void mbwScopeIn(MultiBackendWorkspaceHandle handle);

/**
 * Exit workspace scope
 */
SD_LIB_EXPORT void mbwScopeOut(MultiBackendWorkspaceHandle handle);

/**
 * Check if scope is active
 */
SD_LIB_EXPORT bool mbwIsScopeActive(MultiBackendWorkspaceHandle handle);

/**
 * Get coherence state for a device
 */
SD_LIB_EXPORT int mbwGetCoherenceState(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex);

/**
 * Set coherence state for a device
 */
SD_LIB_EXPORT void mbwSetCoherenceState(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex,
    int state);

/**
 * Mark data as modified on a device
 */
SD_LIB_EXPORT void mbwMarkModified(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex);

/**
 * Transfer data between devices
 */
SD_LIB_EXPORT void mbwTransferTo(
    MultiBackendWorkspaceHandle handle,
    int srcDeviceType, int srcDeviceIndex,
    int dstDeviceType, int dstDeviceIndex);

/**
 * Ensure data is valid on a device
 */
SD_LIB_EXPORT void mbwEnsureValidOn(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex);

/**
 * Get total allocated size
 */
SD_LIB_EXPORT sd::LongType mbwGetTotalAllocatedSize(MultiBackendWorkspaceHandle handle);

/**
 * Get allocated size on a device
 */
SD_LIB_EXPORT sd::LongType mbwGetAllocatedSizeOnDevice(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex);

/**
 * Get current offset
 */
SD_LIB_EXPORT sd::LongType mbwGetCurrentOffset(MultiBackendWorkspaceHandle handle);

/**
 * Release memory on a specific device
 */
SD_LIB_EXPORT void mbwReleaseOnDevice(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex);

/**
 * Synchronize a device
 */
SD_LIB_EXPORT void mbwSyncDevice(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex);

/**
 * Synchronize all devices
 */
SD_LIB_EXPORT void mbwSyncAllDevices(MultiBackendWorkspaceHandle handle);

/**
 * Get number of active devices
 */
SD_LIB_EXPORT int mbwGetActiveDeviceCount(MultiBackendWorkspaceHandle handle);

/**
 * Check if device has allocation
 */
SD_LIB_EXPORT bool mbwHasDeviceAllocation(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex);

/**
 * Get workspace ID
 */
SD_LIB_EXPORT const char* mbwGetId(MultiBackendWorkspaceHandle handle);

}  // extern "C"

}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_MULTI_BACKEND_WORKSPACE_H
