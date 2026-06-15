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

#include <execution/DeviceManager.h>
#include <execution/AffinityManager.h>
#include <system/Environment.h>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace sd {
namespace modelparallel {

// Thread-local current device (function-local to avoid MSVC C2492 with dllexport)
int& DeviceManager::currentDeviceRef() {
    static thread_local int _currentDevice = 0;
    return _currentDevice;
}

DeviceManager& DeviceManager::getInstance() {
    static DeviceManager* instance = nullptr;
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        instance = new DeviceManager();
    });
    return *instance;
}

DeviceManager::DeviceManager() {
    // Will be initialized on first call to initialize()
}

DeviceManager::~DeviceManager() {
    shutdown();
}

bool DeviceManager::initialize() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_initialized.load()) {
        return true;
    }

    // Clear any existing state
    _devices.clear();
    _devicesByType.clear();
    _p2pConnections.clear();

    // Discover all device types
    discoverCpuDevices();
    discoverCudaDevices();
    discoverMetalDevices();
    discoverVulkanDevices();
    discoverOpenClDevices();

    // Initialize P2P connections
    initializeP2PConnections();

    _initialized.store(true);
    return true;
}

void DeviceManager::refresh() {
    std::lock_guard<std::mutex> lock(_mutex);

    // Update memory stats for all devices
    updateAllMemoryStats();
}

void DeviceManager::shutdown() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_initialized.load()) {
        return;
    }

    // Release all device groups
    _deviceGroups.clear();

    // Clear P2P connections
    _p2pConnections.clear();

    // Clear device info
    _devices.clear();
    _devicesByType.clear();

    _initialized.store(false);
}

void DeviceManager::discoverCpuDevices() {
    DeviceInfo cpuInfo;
    cpuInfo.type = DeviceType::CPU;
    cpuInfo.deviceIndex = 0;
    cpuInfo.globalIndex = static_cast<int>(_devices.size());
    cpuInfo.name = "CPU";

    // Get number of hardware threads
    cpuInfo.numCores = static_cast<int>(std::thread::hardware_concurrency());

    // Estimate available memory (this is system-dependent)
    // For now, use a reasonable default
    cpuInfo.totalMemory = 0;  // Will be updated by updateMemoryStats
    cpuInfo.freeMemory = 0;
    cpuInfo.availableMemory = 0;

    cpuInfo.supportsP2P = false;
    cpuInfo.supportsUnifiedMemory = true;
    cpuInfo.supportsAsyncCopy = true;
    cpuInfo.available = true;
    cpuInfo.engine = samediff::ENGINE_CPU;

    _devices.push_back(cpuInfo);
    _devicesByType[DeviceType::CPU].push_back(cpuInfo.globalIndex);
}

// CUDA implementation in execution/cuda/DeviceManager_cuda.cu
#ifndef SD_CUDA
void DeviceManager::discoverCudaDevices() {
    // No CUDA devices to discover in CPU build
}
#endif

void DeviceManager::discoverMetalDevices() {
#ifdef SD_APPLE_MPS
    // Metal device discovery would go here
    // For now, this is a placeholder
#endif
}

void DeviceManager::discoverVulkanDevices() {
    // Vulkan device discovery would go here
    // For now, this is a placeholder
}

void DeviceManager::discoverOpenClDevices() {
    // OpenCL device discovery would go here
    // For now, this is a placeholder
}

#ifndef SD_CUDA
void DeviceManager::initializeP2PConnections() {
    // No P2P connections in CPU build
}
#endif

#ifndef SD_CUDA
void DeviceManager::probeP2PCapabilities(int device1, int device2) {
    // No P2P probing in CPU build
}
#endif

int DeviceManager::getTotalDeviceCount() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return static_cast<int>(_devices.size());
}

int DeviceManager::getDeviceCount(DeviceType type) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _devicesByType.find(type);
    if (it != _devicesByType.end()) {
        return static_cast<int>(it->second.size());
    }
    return 0;
}

int DeviceManager::getCpuCount() const {
    return getDeviceCount(DeviceType::CPU);
}

int DeviceManager::getCudaGpuCount() const {
    return getDeviceCount(DeviceType::CUDA_GPU);
}

std::vector<DeviceInfo> DeviceManager::getAllDevices() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _devices;
}

std::vector<DeviceInfo> DeviceManager::getDevices(DeviceType type) const {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<DeviceInfo> result;

    auto it = _devicesByType.find(type);
    if (it != _devicesByType.end()) {
        for (int idx : it->second) {
            result.push_back(_devices[idx]);
        }
    }
    return result;
}

DeviceInfo DeviceManager::getDeviceInfo(int globalIndex) const {
    std::lock_guard<std::mutex> lock(_mutex);
    if (globalIndex >= 0 && globalIndex < static_cast<int>(_devices.size())) {
        return _devices[globalIndex];
    }
    return DeviceInfo();
}

DeviceInfo DeviceManager::getDeviceInfo(DeviceType type, int localIndex) const {
    int globalIndex = findGlobalIndex(type, localIndex);
    if (globalIndex >= 0) {
        return _devices[globalIndex];
    }
    return DeviceInfo();
}

bool DeviceManager::isDeviceAvailable(DeviceType type, int localIndex) const {
    int globalIndex = findGlobalIndex(type, localIndex);
    if (globalIndex >= 0) {
        return _devices[globalIndex].available;
    }
    return false;
}

bool DeviceManager::isDeviceInUse(int globalIndex) const {
    std::lock_guard<std::mutex> lock(_mutex);
    if (globalIndex >= 0 && globalIndex < static_cast<int>(_devices.size())) {
        return _devices[globalIndex].inUse;
    }
    return false;
}

int DeviceManager::findGlobalIndex(DeviceType type, int localIndex) const {
    auto it = _devicesByType.find(type);
    if (it != _devicesByType.end() && localIndex >= 0 &&
        localIndex < static_cast<int>(it->second.size())) {
        return it->second[localIndex];
    }
    return -1;
}

DeviceInfo* DeviceManager::findDevice(int globalIndex) {
    if (globalIndex >= 0 && globalIndex < static_cast<int>(_devices.size())) {
        return &_devices[globalIndex];
    }
    return nullptr;
}

const DeviceInfo* DeviceManager::findDevice(int globalIndex) const {
    if (globalIndex >= 0 && globalIndex < static_cast<int>(_devices.size())) {
        return &_devices[globalIndex];
    }
    return nullptr;
}

ParallelCapabilities DeviceManager::getParallelCapabilities() const {
    std::lock_guard<std::mutex> lock(_mutex);

    ParallelCapabilities caps;

    int cudaCount = getCudaGpuCount();

    caps.supportsTensorParallel = cudaCount > 1;
    caps.supportsPipelineParallel = cudaCount > 1 || getCpuCount() > 0;
    caps.supportsDataParallel = cudaCount > 1;
    caps.supportsP2P = !_p2pConnections.empty();
    caps.supportsAsyncTransfer = cudaCount > 0;

    caps.maxTensorParallelSize = cudaCount;
    caps.maxPipelineStages = cudaCount + 1;  // GPUs + CPU
    caps.maxDataParallelSize = cudaCount;

    caps.supportedCommPatterns = {
        CommunicationPattern::ALL_REDUCE,
        CommunicationPattern::ALL_GATHER,
        CommunicationPattern::BROADCAST,
        CommunicationPattern::POINT_TO_POINT
    };

    if (caps.supportsP2P) {
        caps.supportedCommPatterns.push_back(CommunicationPattern::RING);
    }

    caps.supportedDeviceTypes = {DeviceType::CPU};
    if (cudaCount > 0) {
        caps.supportedDeviceTypes.push_back(DeviceType::CUDA_GPU);
    }

    return caps;
}

bool DeviceManager::supportsTensorParallel() const {
    return getCudaGpuCount() > 1;
}

bool DeviceManager::supportsPipelineParallel() const {
    return getCudaGpuCount() > 1 || getCpuCount() > 0;
}

bool DeviceManager::supportsP2P(int device1, int device2) const {
    std::lock_guard<std::mutex> lock(_mutex);

    for (const auto& conn : _p2pConnections) {
        if ((conn.sourceDevice == device1 && conn.targetDevice == device2) ||
            (conn.sourceDevice == device2 && conn.targetDevice == device1)) {
            return conn.directAccess;
        }
    }
    return false;
}

P2PConnection DeviceManager::getP2PConnection(int device1, int device2) const {
    std::lock_guard<std::mutex> lock(_mutex);

    for (const auto& conn : _p2pConnections) {
        if ((conn.sourceDevice == device1 && conn.targetDevice == device2) ||
            (conn.sourceDevice == device2 && conn.targetDevice == device1)) {
            return conn;
        }
    }
    return P2PConnection();
}

std::vector<P2PConnection> DeviceManager::getAllP2PConnections() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _p2pConnections;
}

// CUDA implementation in execution/cuda/DeviceManager_cuda.cu
#ifndef SD_CUDA
bool DeviceManager::enableP2P(int device1, int device2) {
    return false;
}
#endif

#ifndef SD_CUDA
void DeviceManager::disableP2P(int device1, int device2) {
    // No P2P in CPU build
}
#endif

void DeviceManager::enableAllP2P() {
    for (const auto& conn : _p2pConnections) {
        enableP2P(conn.sourceDevice, conn.targetDevice);
    }
}

DeviceAllocation DeviceManager::allocateDevices(const ModelParallelConfig& config) {
    std::lock_guard<std::mutex> lock(_mutex);

    DeviceAllocation result;

    for (const auto& spec : config.devices) {
        int globalIdx = findGlobalIndex(spec.type, spec.deviceIndex);
        if (globalIdx < 0) {
            result.success = false;
            result.errorMessage = "Device not found: " + deviceTypeToString(spec.type) +
                                  " index " + std::to_string(spec.deviceIndex);
            return result;
        }

        if (_devices[globalIdx].inUse) {
            result.success = false;
            result.errorMessage = "Device already in use: " + _devices[globalIdx].name;
            return result;
        }

        result.allocatedDevices.push_back(_devices[globalIdx]);
    }

    // Mark devices as in use
    for (auto& device : result.allocatedDevices) {
        int idx = device.globalIndex;
        _devices[idx].inUse = true;
        _devices[idx].currentUserCount++;
    }

    result.success = true;
    return result;
}

DeviceAllocation DeviceManager::allocateDevices(DeviceType type, int count) {
    std::lock_guard<std::mutex> lock(_mutex);

    DeviceAllocation result;

    auto it = _devicesByType.find(type);
    if (it == _devicesByType.end() || static_cast<int>(it->second.size()) < count) {
        result.success = false;
        result.errorMessage = "Not enough devices of type " + deviceTypeToString(type);
        return result;
    }

    int allocated = 0;
    for (int idx : it->second) {
        if (!_devices[idx].inUse && allocated < count) {
            result.allocatedDevices.push_back(_devices[idx]);
            _devices[idx].inUse = true;
            _devices[idx].currentUserCount++;
            allocated++;
        }
    }

    if (allocated < count) {
        // Rollback
        for (auto& device : result.allocatedDevices) {
            _devices[device.globalIndex].inUse = false;
            _devices[device.globalIndex].currentUserCount--;
        }
        result.allocatedDevices.clear();
        result.success = false;
        result.errorMessage = "Not enough available devices";
        return result;
    }

    result.success = true;
    return result;
}

void DeviceManager::acquireDevice(int globalIndex) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (auto* device = findDevice(globalIndex)) {
        device->inUse = true;
        device->currentUserCount++;
    }
}

void DeviceManager::releaseDevice(int globalIndex) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (auto* device = findDevice(globalIndex)) {
        device->currentUserCount--;
        if (device->currentUserCount <= 0) {
            device->inUse = false;
            device->currentUserCount = 0;
        }
    }
}

void DeviceManager::releaseAllDevices() {
    std::lock_guard<std::mutex> lock(_mutex);

    for (auto& device : _devices) {
        device.inUse = false;
        device.currentUserCount = 0;
    }
}

size_t DeviceManager::getTotalMemory(DeviceType type) const {
    std::lock_guard<std::mutex> lock(_mutex);

    size_t total = 0;
    auto it = _devicesByType.find(type);
    if (it != _devicesByType.end()) {
        for (int idx : it->second) {
            total += _devices[idx].totalMemory;
        }
    }
    return total;
}

size_t DeviceManager::getFreeMemory(DeviceType type) const {
    std::lock_guard<std::mutex> lock(_mutex);

    size_t total = 0;
    auto it = _devicesByType.find(type);
    if (it != _devicesByType.end()) {
        for (int idx : it->second) {
            total += _devices[idx].freeMemory;
        }
    }
    return total;
}

size_t DeviceManager::getDeviceFreeMemory(int globalIndex) const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (const auto* device = findDevice(globalIndex)) {
        return device->freeMemory;
    }
    return 0;
}

#ifndef SD_CUDA
void DeviceManager::updateMemoryStats(int globalIndex) {
    // CPU build: no GPU memory stats to update
}
#endif

void DeviceManager::updateAllMemoryStats() {
    for (size_t i = 0; i < _devices.size(); ++i) {
        updateMemoryStats(static_cast<int>(i));
    }
}

bool DeviceManager::reserveMemory(int globalIndex, size_t bytes) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto* device = findDevice(globalIndex);
    if (device == nullptr || !device->available) {
        return false;
    }

    updateMemoryStats(globalIndex);

    size_t reserved = 0;
    auto it = _reservedMemory.find(globalIndex);
    if (it != _reservedMemory.end()) {
        reserved = it->second;
    }

    if (bytes == 0) {
        return true;
    }

    if (device->freeMemory < reserved + bytes) {
        return false;
    }

    _reservedMemory[globalIndex] = reserved + bytes;
    device->availableMemory = (device->freeMemory > _reservedMemory[globalIndex])
        ? (device->freeMemory - _reservedMemory[globalIndex])
        : 0;

    return true;
}

void DeviceManager::releaseReservedMemory(int globalIndex, size_t bytes) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto* device = findDevice(globalIndex);
    if (device == nullptr) {
        return;
    }

    auto it = _reservedMemory.find(globalIndex);
    if (it == _reservedMemory.end()) {
        device->availableMemory = device->freeMemory;
        return;
    }

    if (bytes >= it->second) {
        _reservedMemory.erase(it);
        device->availableMemory = device->freeMemory;
        return;
    }

    it->second -= bytes;
    device->availableMemory = (device->freeMemory > it->second) ? (device->freeMemory - it->second) : 0;
}

std::vector<DeviceInfo> DeviceManager::findBestDevices(size_t requiredMemory, int count, bool preferP2P) {
    if (count <= 0) {
        return {};
    }

    if (!_initialized.load()) {
        initialize();
    }

    std::lock_guard<std::mutex> lock(_mutex);
    updateAllMemoryStats();

    std::vector<DeviceInfo> candidates;
    candidates.reserve(_devices.size());

    for (const auto& device : _devices) {
        if (!device.available || device.inUse) {
            continue;
        }

        if (requiredMemory > 0 && device.availableMemory < requiredMemory) {
            continue;
        }

        candidates.push_back(device);
    }

    if (candidates.empty()) {
        return {};
    }

    std::sort(candidates.begin(), candidates.end(), [preferP2P](const DeviceInfo& a, const DeviceInfo& b) {
        if (preferP2P && a.supportsP2P != b.supportsP2P) {
            return a.supportsP2P > b.supportsP2P;
        }
        if (a.availableMemory != b.availableMemory) {
            return a.availableMemory > b.availableMemory;
        }
        return a.globalIndex < b.globalIndex;
    });

    if (count >= static_cast<int>(candidates.size())) {
        return candidates;
    }

    return std::vector<DeviceInfo>(candidates.begin(), candidates.begin() + count);
}

DeviceInfo DeviceManager::getBestGpu() const {
    auto* self = const_cast<DeviceManager*>(this);
    if (!self->_initialized.load()) {
        self->initialize();
    }

    std::lock_guard<std::mutex> lock(_mutex);
    self->updateAllMemoryStats();

    bool found = false;
    DeviceInfo best;

    for (const auto& device : _devices) {
        if (device.type != DeviceType::CUDA_GPU) {
            continue;
        }
        if (!device.available || device.inUse) {
            continue;
        }

        if (!found || device.availableMemory > best.availableMemory) {
            best = device;
            found = true;
        }
    }

    if (found) {
        return best;
    }

    for (const auto& device : _devices) {
        if (device.type == DeviceType::CUDA_GPU) {
            return device;
        }
    }

    if (!_devices.empty()) {
        return _devices.front();
    }

    return DeviceInfo{};
}

void DeviceManager::setCurrentDevice(int globalIndex) {
    if (globalIndex < 0 || globalIndex >= static_cast<int>(_devices.size())) {
        return;
    }

    const auto& device = _devices[globalIndex];
    setCurrentDeviceCuda(device);
    currentDeviceRef() = globalIndex;
}

#ifndef SD_CUDA
void DeviceManager::setCurrentDeviceCuda(const DeviceInfo& device) {
    // No CUDA device switching in CPU build
}
#endif

int DeviceManager::getCurrentDevice() const {
    return currentDeviceRef();
}

samediff::Engine DeviceManager::deviceTypeToEngine(DeviceType type) {
    switch (type) {
        case DeviceType::CUDA_GPU:
            return samediff::ENGINE_CUDA;
        case DeviceType::TPU:
            return samediff::ENGINE_TPU;
        case DeviceType::CPU:
        default:
            return samediff::ENGINE_CPU;
    }
}

DeviceType DeviceManager::engineToDeviceType(samediff::Engine engine) {
    switch (engine) {
        case samediff::ENGINE_CUDA:
            return DeviceType::CUDA_GPU;
        case samediff::ENGINE_TPU:
            return DeviceType::TPU;
        case samediff::ENGINE_CPU:
        default:
            return DeviceType::CPU;
    }
}

int DeviceManager::getGlobalIndex(samediff::Engine engine, int localIndex) const {
    DeviceType type = engineToDeviceType(engine);
    return findGlobalIndex(type, localIndex);
}

#ifndef SD_CUDA
void DeviceManager::synchronizeAll() {
    // No GPU synchronization in CPU build
}

void DeviceManager::synchronize(int globalIndex) {
    // No GPU synchronization in CPU build
}
#endif

std::string DeviceManager::deviceToString(const DeviceInfo& device) const {
    std::ostringstream ss;
    ss << "[" << device.globalIndex << "] "
       << deviceTypeToString(device.type) << ":" << device.deviceIndex
       << " \"" << device.name << "\" "
       << (device.totalMemory / (1024*1024)) << "MB";
    if (device.inUse) {
        ss << " (in use)";
    }
    return ss.str();
}

void DeviceManager::printDeviceInfo() const {
    std::lock_guard<std::mutex> lock(_mutex);

    if (sd::Environment::getInstance().isVerbose()) {
        std::cout << "=== Device Manager ===" << std::endl;
        std::cout << "Total devices: " << _devices.size() << std::endl;

        for (const auto& device : _devices) {
            std::cout << "  " << deviceToString(device) << std::endl;
        }

        std::cout << "P2P connections: " << _p2pConnections.size() << std::endl;
        for (const auto& conn : _p2pConnections) {
            std::cout << "  " << conn.sourceDevice << " <-> " << conn.targetDevice;
            if (conn.nvlinkConnected) {
                std::cout << " (NVLink)";
            }
            std::cout << std::endl;
        }
    }
}

// DeviceContextGuard implementation
DeviceContextGuard::DeviceContextGuard(int globalIndex)
    : _deviceIndex(globalIndex)
    , _previousDevice(DeviceManager::getInstance().getCurrentDevice()) {
    DeviceManager::getInstance().setCurrentDevice(globalIndex);
}

DeviceContextGuard::DeviceContextGuard(DeviceType type, int localIndex)
    : _previousDevice(DeviceManager::getInstance().getCurrentDevice()) {
    auto& dm = DeviceManager::getInstance();
    auto info = dm.getDeviceInfo(type, localIndex);
    _deviceIndex = info.globalIndex;
    dm.setCurrentDevice(_deviceIndex);
}

DeviceContextGuard::~DeviceContextGuard() {
    DeviceManager::getInstance().setCurrentDevice(_previousDevice);
}

}  // namespace modelparallel
}  // namespace sd
