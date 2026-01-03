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

#include <memory/MultiBackendWorkspace.h>
#include <helpers/logger.h>
#include <stdexcept>
#include <cstring>

namespace sd {
namespace memory {

// ========================
// Constructors / Destructor
// ========================

MultiBackendWorkspace::MultiBackendWorkspace(const MultiBackendWorkspaceConfig& config,
                                             const std::string& id)
    : _id(id.empty() ? "mbw_" + std::to_string(reinterpret_cast<uintptr_t>(this)) : id),
      _config(config),
      _primaryDevice(config.primaryDevice),
      _globalVersion(0),
      _scopeDepth(0),
      _scopeActive(false),
      _totalAllocations(0),
      _totalDeallocations(0),
      _totalTransfers(0) {

    // Initialize primary device workspace if initial size > 0
    if (config.initialSize > 0) {
        initDeviceWorkspace(_primaryDevice, config.initialSize);
    }
}

MultiBackendWorkspace::MultiBackendWorkspace(sd::LongType initialSize,
                                             const DeviceDescriptor& primaryDevice)
    : _id("mbw_" + std::to_string(reinterpret_cast<uintptr_t>(this))),
      _primaryDevice(primaryDevice),
      _globalVersion(0),
      _scopeDepth(0),
      _scopeActive(false),
      _totalAllocations(0),
      _totalDeallocations(0),
      _totalTransfers(0) {

    _config.initialSize = initialSize;
    _config.primaryDevice = primaryDevice;

    if (initialSize > 0) {
        initDeviceWorkspace(_primaryDevice, initialSize);
    }
}

MultiBackendWorkspace::~MultiBackendWorkspace() {
    destroy();
}

MultiBackendWorkspace::MultiBackendWorkspace(MultiBackendWorkspace&& other) noexcept
    : _id(std::move(other._id)),
      _config(std::move(other._config)),
      _deviceAllocations(std::move(other._deviceAllocations)),
      _primaryDevice(std::move(other._primaryDevice)),
      _globalVersion(other._globalVersion.load()),
      _scopeDepth(other._scopeDepth.load()),
      _scopeActive(other._scopeActive.load()),
      _totalAllocations(other._totalAllocations.load()),
      _totalDeallocations(other._totalDeallocations.load()),
      _totalTransfers(other._totalTransfers.load()) {
    // Clear the source
    other._deviceAllocations.clear();
}

MultiBackendWorkspace& MultiBackendWorkspace::operator=(MultiBackendWorkspace&& other) noexcept {
    if (this != &other) {
        // Clean up current state
        destroy();

        // Move from other
        _id = std::move(other._id);
        _config = std::move(other._config);
        _deviceAllocations = std::move(other._deviceAllocations);
        _primaryDevice = std::move(other._primaryDevice);
        _globalVersion = other._globalVersion.load();
        _scopeDepth = other._scopeDepth.load();
        _scopeActive = other._scopeActive.load();
        _totalAllocations = other._totalAllocations.load();
        _totalDeallocations = other._totalDeallocations.load();
        _totalTransfers = other._totalTransfers.load();

        other._deviceAllocations.clear();
    }
    return *this;
}

// ========================
// Internal Helpers
// ========================

void MultiBackendWorkspace::initDeviceWorkspace(const DeviceDescriptor& device, sd::LongType size) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end() && it->second.workspace != nullptr) {
        // Already exists, expand if needed
        if (it->second.workspace->getCurrentSize() < size) {
            it->second.workspace->expandTo(size, 0);
        }
        return;
    }

    // Create new workspace
    DeviceAllocation allocation;
    allocation.workspace = new Workspace(size, 0);
    allocation.coherenceState = CoherenceState::EXCLUSIVE;
    allocation.version = _globalVersion.load();
    allocation.isOwned = true;

    _deviceAllocations[device] = allocation;
}

void MultiBackendWorkspace::freeDeviceWorkspace(const DeviceDescriptor& device) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end()) {
        if (it->second.isOwned && it->second.workspace != nullptr) {
            delete it->second.workspace;
            _totalDeallocations++;
        }
        _deviceAllocations.erase(it);
    }
}

void MultiBackendWorkspace::invalidateOtherDevices(const DeviceDescriptor& exceptDevice) {
    for (auto& pair : _deviceAllocations) {
        if (pair.first != exceptDevice) {
            pair.second.coherenceState = CoherenceState::INVALID;
        }
    }
}

// ========================
// Device Management
// ========================

void MultiBackendWorkspace::setPrimaryDevice(const DeviceDescriptor& device) {
    std::lock_guard<std::mutex> lock(_mutex);
    _primaryDevice = device;
}

std::vector<DeviceDescriptor> MultiBackendWorkspace::getActiveDevices() const {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<DeviceDescriptor> devices;
    devices.reserve(_deviceAllocations.size());
    for (const auto& pair : _deviceAllocations) {
        devices.push_back(pair.first);
    }
    return devices;
}

bool MultiBackendWorkspace::hasDeviceAllocation(const DeviceDescriptor& device) const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _deviceAllocations.find(device) != _deviceAllocations.end();
}

// ========================
// Allocation
// ========================

void* MultiBackendWorkspace::allocateBytes(sd::LongType numBytes) {
    return allocateBytes(_primaryDevice, numBytes);
}

void* MultiBackendWorkspace::allocateBytes(const DeviceDescriptor& device, sd::LongType numBytes) {
    return allocateBytes(device, MemoryType::HOST, numBytes);
}

void* MultiBackendWorkspace::allocateBytes(const DeviceDescriptor& device, MemoryType type,
                                            sd::LongType numBytes) {
    if (numBytes <= 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(_mutex);

    // Ensure workspace exists on device
    auto it = _deviceAllocations.find(device);
    if (it == _deviceAllocations.end() || it->second.workspace == nullptr) {
        // Create workspace with at least the requested size
        sd::LongType size = std::max(numBytes, _config.initialSize);
        initDeviceWorkspace(device, size);
        it = _deviceAllocations.find(device);
    }

    // Allocate from workspace
    void* ptr = it->second.workspace->allocateBytes(type, numBytes);
    _totalAllocations++;

    // Mark this device as having the exclusive copy
    it->second.coherenceState = CoherenceState::EXCLUSIVE;
    it->second.version = ++_globalVersion;

    // Invalidate other devices if mirroring is not enabled
    if (!_config.crossDeviceMirroring) {
        invalidateOtherDevices(device);
    }

    return ptr;
}

Workspace* MultiBackendWorkspace::getDeviceWorkspace(const DeviceDescriptor& device) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end()) {
        return it->second.workspace;
    }
    return nullptr;
}

void MultiBackendWorkspace::ensureWorkspaceOnDevice(const DeviceDescriptor& device,
                                                     sd::LongType minSize) {
    sd::LongType size = std::max(minSize, _config.initialSize);
    initDeviceWorkspace(device, size);
}

// ========================
// Scope Management
// ========================

void MultiBackendWorkspace::scopeIn() {
    std::lock_guard<std::mutex> lock(_mutex);

    _scopeDepth++;
    _scopeActive = true;

    // Call scopeIn on all device workspaces
    for (auto& pair : _deviceAllocations) {
        if (pair.second.workspace != nullptr) {
            pair.second.workspace->scopeIn();
        }
    }
}

void MultiBackendWorkspace::scopeOut() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_scopeDepth > 0) {
        _scopeDepth--;
    }

    if (_scopeDepth == 0) {
        _scopeActive = false;
    }

    // Call scopeOut on all device workspaces
    for (auto& pair : _deviceAllocations) {
        if (pair.second.workspace != nullptr) {
            pair.second.workspace->scopeOut();
        }
    }
}

// ========================
// Coherence Management
// ========================

CoherenceState MultiBackendWorkspace::getCoherenceState(const DeviceDescriptor& device) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end()) {
        return it->second.coherenceState;
    }
    return CoherenceState::INVALID;
}

void MultiBackendWorkspace::setCoherenceState(const DeviceDescriptor& device,
                                               CoherenceState state) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end()) {
        it->second.coherenceState = state;
    }
}

void MultiBackendWorkspace::markModified(const DeviceDescriptor& device) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end()) {
        it->second.coherenceState = CoherenceState::MODIFIED;
        it->second.version = ++_globalVersion;
        invalidateOtherDevices(device);
    }
}

void MultiBackendWorkspace::invalidateDevice(const DeviceDescriptor& device) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end()) {
        it->second.coherenceState = CoherenceState::INVALID;
    }
}

void MultiBackendWorkspace::invalidateAllExcept(const DeviceDescriptor& device) {
    std::lock_guard<std::mutex> lock(_mutex);
    invalidateOtherDevices(device);
}

sd::LongType MultiBackendWorkspace::getDeviceVersion(const DeviceDescriptor& device) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end()) {
        return it->second.version;
    }
    return 0;
}

// ========================
// Transfers
// ========================

void MultiBackendWorkspace::transferTo(const DeviceDescriptor& source,
                                        const DeviceDescriptor& target) {
    std::lock_guard<std::mutex> lock(_transferMutex);

    auto srcIt = _deviceAllocations.find(source);
    if (srcIt == _deviceAllocations.end() || srcIt->second.workspace == nullptr) {
        throw std::runtime_error("Source device has no workspace allocation");
    }

    // Ensure target workspace exists
    ensureWorkspaceOnDevice(target, srcIt->second.workspace->getCurrentSize());

    auto tgtIt = _deviceAllocations.find(target);

    // For CPU-only implementation, we do a memcpy
    // In CUDA implementation, this would be cudaMemcpy
    Workspace* srcWs = srcIt->second.workspace;
    Workspace* tgtWs = tgtIt->second.workspace;

    // Note: Actual data transfer would require access to the raw pointers
    // This is a simplified version - full implementation would track allocations

    // Update coherence state
    tgtIt->second.coherenceState = CoherenceState::SHARED;
    tgtIt->second.version = srcIt->second.version;
    srcIt->second.coherenceState = CoherenceState::SHARED;

    _totalTransfers++;
}

void MultiBackendWorkspace::ensureValidOn(const DeviceDescriptor& device) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end() &&
        it->second.coherenceState != CoherenceState::INVALID) {
        return;  // Already valid
    }

    // Find a valid source device
    for (const auto& pair : _deviceAllocations) {
        if (pair.first != device &&
            pair.second.coherenceState != CoherenceState::INVALID) {
            // Transfer from this device
            transferTo(pair.first, device);
            return;
        }
    }

    // No valid source found, initialize fresh
    ensureWorkspaceOnDevice(device, _config.initialSize);
}

void MultiBackendWorkspace::syncDevice(const DeviceDescriptor& device) {
    // For CPU, this is a no-op
    // CUDA implementation would call cudaDeviceSynchronize or stream sync
}

void MultiBackendWorkspace::syncAllDevices() {
    for (const auto& pair : _deviceAllocations) {
        syncDevice(pair.first);
    }
}

// ========================
// Size and Statistics
// ========================

sd::LongType MultiBackendWorkspace::getTotalAllocatedSize() const {
    std::lock_guard<std::mutex> lock(_mutex);
    sd::LongType total = 0;
    for (const auto& pair : _deviceAllocations) {
        if (pair.second.workspace != nullptr) {
            total += pair.second.workspace->getAllocatedSize();
        }
    }
    return total;
}

sd::LongType MultiBackendWorkspace::getAllocatedSizeOnDevice(const DeviceDescriptor& device) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end() && it->second.workspace != nullptr) {
        return it->second.workspace->getAllocatedSize();
    }
    return 0;
}

sd::LongType MultiBackendWorkspace::getCurrentOffset() const {
    return getCurrentOffsetOnDevice(_primaryDevice);
}

sd::LongType MultiBackendWorkspace::getCurrentOffsetOnDevice(const DeviceDescriptor& device) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end() && it->second.workspace != nullptr) {
        return it->second.workspace->getCurrentOffset();
    }
    return 0;
}

sd::LongType MultiBackendWorkspace::getSpilledSize() const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(_primaryDevice);
    if (it != _deviceAllocations.end() && it->second.workspace != nullptr) {
        return it->second.workspace->getSpilledSize();
    }
    return 0;
}

// ========================
// Cleanup
// ========================

void MultiBackendWorkspace::destroy() {
    std::lock_guard<std::mutex> lock(_mutex);

    for (auto& pair : _deviceAllocations) {
        if (pair.second.isOwned && pair.second.workspace != nullptr) {
            delete pair.second.workspace;
            pair.second.workspace = nullptr;
            _totalDeallocations++;
        }
    }
    _deviceAllocations.clear();
    _scopeActive = false;
    _scopeDepth = 0;
}

void MultiBackendWorkspace::releaseOnDevice(const DeviceDescriptor& device) {
    freeDeviceWorkspace(device);
}

void MultiBackendWorkspace::resetStatistics() {
    _totalAllocations = 0;
    _totalDeallocations = 0;
    _totalTransfers = 0;
}

MultiBackendWorkspace* MultiBackendWorkspace::clone() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return new MultiBackendWorkspace(_config, _id + "_clone");
}

// ========================
// C-style API Implementation
// ========================

extern "C" {

MultiBackendWorkspaceHandle createMultiBackendWorkspace(
    sd::LongType initialSize,
    int primaryDeviceType,
    int primaryDeviceIndex) {

    DeviceDescriptor device(static_cast<DeviceType>(primaryDeviceType), primaryDeviceIndex);
    return new MultiBackendWorkspace(initialSize, device);
}

MultiBackendWorkspaceHandle createMultiBackendWorkspaceWithConfig(
    sd::LongType initialSize,
    sd::LongType maxSize,
    bool crossDeviceMirroring,
    bool asyncTransfers,
    int primaryDeviceType,
    int primaryDeviceIndex,
    const char* id) {

    MultiBackendWorkspaceConfig config;
    config.initialSize = initialSize;
    config.maxSize = maxSize;
    config.crossDeviceMirroring = crossDeviceMirroring;
    config.asyncTransfers = asyncTransfers;
    config.primaryDevice = DeviceDescriptor(static_cast<DeviceType>(primaryDeviceType),
                                             primaryDeviceIndex);

    std::string idStr = (id != nullptr) ? id : "";
    return new MultiBackendWorkspace(config, idStr);
}

void destroyMultiBackendWorkspace(MultiBackendWorkspaceHandle handle) {
    if (handle != nullptr) {
        handle->destroy();
        delete handle;
    }
}

void* mbwAllocateBytes(MultiBackendWorkspaceHandle handle, sd::LongType numBytes) {
    if (handle == nullptr) return nullptr;
    return handle->allocateBytes(numBytes);
}

void* mbwAllocateBytesOnDevice(
    MultiBackendWorkspaceHandle handle,
    sd::LongType numBytes,
    int deviceType,
    int deviceIndex) {

    if (handle == nullptr) return nullptr;
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    return handle->allocateBytes(device, numBytes);
}

void mbwScopeIn(MultiBackendWorkspaceHandle handle) {
    if (handle != nullptr) {
        handle->scopeIn();
    }
}

void mbwScopeOut(MultiBackendWorkspaceHandle handle) {
    if (handle != nullptr) {
        handle->scopeOut();
    }
}

bool mbwIsScopeActive(MultiBackendWorkspaceHandle handle) {
    if (handle == nullptr) return false;
    return handle->isScopeActive();
}

int mbwGetCoherenceState(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex) {

    if (handle == nullptr) return static_cast<int>(CoherenceState::INVALID);
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    return static_cast<int>(handle->getCoherenceState(device));
}

void mbwSetCoherenceState(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex,
    int state) {

    if (handle == nullptr) return;
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    handle->setCoherenceState(device, static_cast<CoherenceState>(state));
}

void mbwMarkModified(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex) {

    if (handle == nullptr) return;
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    handle->markModified(device);
}

void mbwTransferTo(
    MultiBackendWorkspaceHandle handle,
    int srcDeviceType, int srcDeviceIndex,
    int dstDeviceType, int dstDeviceIndex) {

    if (handle == nullptr) return;
    DeviceDescriptor src(static_cast<DeviceType>(srcDeviceType), srcDeviceIndex);
    DeviceDescriptor dst(static_cast<DeviceType>(dstDeviceType), dstDeviceIndex);
    handle->transferTo(src, dst);
}

void mbwEnsureValidOn(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex) {

    if (handle == nullptr) return;
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    handle->ensureValidOn(device);
}

sd::LongType mbwGetTotalAllocatedSize(MultiBackendWorkspaceHandle handle) {
    if (handle == nullptr) return 0;
    return handle->getTotalAllocatedSize();
}

sd::LongType mbwGetAllocatedSizeOnDevice(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex) {

    if (handle == nullptr) return 0;
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    return handle->getAllocatedSizeOnDevice(device);
}

sd::LongType mbwGetCurrentOffset(MultiBackendWorkspaceHandle handle) {
    if (handle == nullptr) return 0;
    return handle->getCurrentOffset();
}

void mbwReleaseOnDevice(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex) {

    if (handle == nullptr) return;
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    handle->releaseOnDevice(device);
}

void mbwSyncDevice(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex) {

    if (handle == nullptr) return;
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    handle->syncDevice(device);
}

void mbwSyncAllDevices(MultiBackendWorkspaceHandle handle) {
    if (handle != nullptr) {
        handle->syncAllDevices();
    }
}

int mbwGetActiveDeviceCount(MultiBackendWorkspaceHandle handle) {
    if (handle == nullptr) return 0;
    return static_cast<int>(handle->getActiveDevices().size());
}

bool mbwHasDeviceAllocation(
    MultiBackendWorkspaceHandle handle,
    int deviceType,
    int deviceIndex) {

    if (handle == nullptr) return false;
    DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
    return handle->hasDeviceAllocation(device);
}

const char* mbwGetId(MultiBackendWorkspaceHandle handle) {
    if (handle == nullptr) return "";
    return handle->getId().c_str();
}

}  // extern "C"

}  // namespace memory
}  // namespace sd
