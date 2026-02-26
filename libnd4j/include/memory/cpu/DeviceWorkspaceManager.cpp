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

#include <memory/DeviceWorkspaceManager.h>
#include <helpers/logger.h>
#include <sstream>
#include <iomanip>

namespace sd {
namespace memory {

// Static members
DeviceWorkspaceManager* DeviceWorkspaceManager::_instance = nullptr;
std::mutex DeviceWorkspaceManager::_instanceMutex;

// ========================
// Constructor / Destructor
// ========================

DeviceWorkspaceManager::DeviceWorkspaceManager()
    : _totalWorkspacesCreated(0),
      _totalWorkspacesDestroyed(0),
      _totalMemoryAllocated(0) {

    // Set default configuration
    _defaultConfig.initialSize = 0;
    _defaultConfig.maxSize = LLONG_MAX;
    _defaultConfig.crossDeviceMirroring = false;
    _defaultConfig.asyncTransfers = true;
    _defaultConfig.primaryDevice = DeviceDescriptor(DeviceType::CPU, 0);
}

DeviceWorkspaceManager::~DeviceWorkspaceManager() {
    destroyAllWorkspaces();
}

// ========================
// Singleton Management
// ========================

DeviceWorkspaceManager* DeviceWorkspaceManager::getInstance() {
    if (_instance == nullptr) {
        std::lock_guard<std::mutex> lock(_instanceMutex);
        if (_instance == nullptr) {
            _instance = new DeviceWorkspaceManager();
        }
    }
    return _instance;
}

void DeviceWorkspaceManager::shutdown() {
    std::lock_guard<std::mutex> lock(_instanceMutex);
    if (_instance != nullptr) {
        delete _instance;
        _instance = nullptr;
    }
}

// ========================
// Internal Helpers
// ========================

ThreadWorkspaceEntry* DeviceWorkspaceManager::getThreadEntry() {
    std::thread::id threadId = std::this_thread::get_id();

    // Try read lock first
    {
        std::lock_guard<std::mutex> readLock(_threadWorkspacesMutex);
        auto it = _threadWorkspaces.find(threadId);
        if (it != _threadWorkspaces.end()) {
            return it->second.get();
        }
    }

    // Need to create - upgrade to write lock
    std::unique_lock<std::mutex> writeLock(_threadWorkspacesMutex);
    auto it = _threadWorkspaces.find(threadId);
    if (it != _threadWorkspaces.end()) {
        return it->second.get();
    }

    // Create new entry
    auto entry = std::make_unique<ThreadWorkspaceEntry>();
    ThreadWorkspaceEntry* ptr = entry.get();
    _threadWorkspaces[threadId] = std::move(entry);
    return ptr;
}

// ========================
// Workspace Creation
// ========================

MultiBackendWorkspace* DeviceWorkspaceManager::createWorkspace(
    const MultiBackendWorkspaceConfig& config,
    const std::string& id) {

    std::string wsId = id;
    if (wsId.empty()) {
        // Generate unique ID
        std::ostringstream oss;
        oss << "ws_" << std::this_thread::get_id() << "_" << _totalWorkspacesCreated.load();
        wsId = oss.str();
    }

    MultiBackendWorkspace* workspace = new MultiBackendWorkspace(config, wsId);

    // Register in thread-local storage
    ThreadWorkspaceEntry* entry = getThreadEntry();
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        entry->workspaces[wsId] = workspace;
    }

    _totalWorkspacesCreated++;
    _totalMemoryAllocated += config.initialSize;

    return workspace;
}

MultiBackendWorkspace* DeviceWorkspaceManager::createWorkspace(
    sd::LongType initialSize,
    const std::string& id) {

    MultiBackendWorkspaceConfig config = _defaultConfig;
    config.initialSize = initialSize;
    return createWorkspace(config, id);
}

MultiBackendWorkspace* DeviceWorkspaceManager::createWorkspaceOnDevice(
    const MultiBackendWorkspaceConfig& config,
    const std::string& id,
    const DeviceDescriptor& device) {

    MultiBackendWorkspaceConfig deviceConfig = config;
    deviceConfig.primaryDevice = device;
    return createWorkspace(deviceConfig, id);
}

// ========================
// Workspace Retrieval
// ========================

MultiBackendWorkspace* DeviceWorkspaceManager::getWorkspace(const std::string& id) {
    ThreadWorkspaceEntry* entry = getThreadEntry();
    std::lock_guard<std::mutex> lock(entry->mutex);

    auto it = entry->workspaces.find(id);
    if (it != entry->workspaces.end()) {
        return it->second;
    }
    return nullptr;
}

MultiBackendWorkspace* DeviceWorkspaceManager::getOrCreateWorkspace(
    const MultiBackendWorkspaceConfig& config,
    const std::string& id) {

    MultiBackendWorkspace* workspace = getWorkspace(id);
    if (workspace == nullptr) {
        workspace = createWorkspace(config, id);
    }
    return workspace;
}

MultiBackendWorkspace* DeviceWorkspaceManager::getAndActivateWorkspace(
    const MultiBackendWorkspaceConfig& config,
    const std::string& id) {

    MultiBackendWorkspace* workspace = getOrCreateWorkspace(config, id);
    workspace->scopeIn();
    return workspace;
}

MultiBackendWorkspace* DeviceWorkspaceManager::getAndActivateWorkspace(const std::string& id) {
    return getAndActivateWorkspace(_defaultConfig, id);
}

bool DeviceWorkspaceManager::workspaceExists(const std::string& id) {
    return getWorkspace(id) != nullptr;
}

// ========================
// Workspace Destruction
// ========================

void DeviceWorkspaceManager::destroyWorkspace(MultiBackendWorkspace* workspace) {
    if (workspace == nullptr) return;

    std::string id = workspace->getId();

    // Remove from thread-local storage
    ThreadWorkspaceEntry* entry = getThreadEntry();
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        entry->workspaces.erase(id);
    }

    // Remove from global registry if present
    {
        std::lock_guard<std::mutex> lock(_globalWorkspacesMutex);
        _globalWorkspaces.erase(id);
    }

    // Track memory before destruction
    sd::LongType memoryFreed = workspace->getTotalAllocatedSize();

    // Destroy the workspace
    workspace->destroy();
    delete workspace;

    _totalWorkspacesDestroyed++;
    _totalMemoryAllocated -= memoryFreed;
}

void DeviceWorkspaceManager::destroyWorkspace(const std::string& id) {
    MultiBackendWorkspace* workspace = getWorkspace(id);
    if (workspace != nullptr) {
        destroyWorkspace(workspace);
    }
}

void DeviceWorkspaceManager::destroyAllWorkspacesForCurrentThread() {
    ThreadWorkspaceEntry* entry = getThreadEntry();

    std::vector<MultiBackendWorkspace*> toDestroy;
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        for (auto& pair : entry->workspaces) {
            toDestroy.push_back(pair.second);
        }
        entry->workspaces.clear();
    }

    for (auto* workspace : toDestroy) {
        // Remove from global if present
        {
            std::lock_guard<std::mutex> lock(_globalWorkspacesMutex);
            _globalWorkspaces.erase(workspace->getId());
        }

        sd::LongType memoryFreed = workspace->getTotalAllocatedSize();
        workspace->destroy();
        delete workspace;

        _totalWorkspacesDestroyed++;
        _totalMemoryAllocated -= memoryFreed;
    }
}

void DeviceWorkspaceManager::destroyAllWorkspaces() {
    // First, clear global registry
    {
        std::lock_guard<std::mutex> lock(_globalWorkspacesMutex);
        _globalWorkspaces.clear();
    }

    // Then destroy all thread-local workspaces
    std::unique_lock<std::mutex> writeLock(_threadWorkspacesMutex);

    for (auto& threadPair : _threadWorkspaces) {
        std::lock_guard<std::mutex> lock(threadPair.second->mutex);
        for (auto& wsPair : threadPair.second->workspaces) {
            if (wsPair.second != nullptr) {
                wsPair.second->destroy();
                delete wsPair.second;
                _totalWorkspacesDestroyed++;
            }
        }
        threadPair.second->workspaces.clear();
    }
    _threadWorkspaces.clear();
    _totalMemoryAllocated = 0;
}

// ========================
// Global Workspace Registry
// ========================

void DeviceWorkspaceManager::makeGlobal(MultiBackendWorkspace* workspace) {
    if (workspace == nullptr) return;

    std::lock_guard<std::mutex> lock(_globalWorkspacesMutex);
    _globalWorkspaces[workspace->getId()] = workspace;
}

MultiBackendWorkspace* DeviceWorkspaceManager::getGlobalWorkspace(const std::string& id) {
    std::lock_guard<std::mutex> lock(_globalWorkspacesMutex);
    auto it = _globalWorkspaces.find(id);
    if (it != _globalWorkspaces.end()) {
        return it->second;
    }
    return nullptr;
}

void DeviceWorkspaceManager::removeFromGlobal(const std::string& id) {
    std::lock_guard<std::mutex> lock(_globalWorkspacesMutex);
    _globalWorkspaces.erase(id);
}

// ========================
// Configuration
// ========================

void DeviceWorkspaceManager::setDefaultConfiguration(const MultiBackendWorkspaceConfig& config) {
    std::lock_guard<std::mutex> lock(_configMutex);
    _defaultConfig = config;
}

MultiBackendWorkspaceConfig DeviceWorkspaceManager::getDefaultConfiguration() {
    std::lock_guard<std::mutex> lock(_configMutex);
    return _defaultConfig;
}

// ========================
// Statistics and Info
// ========================

std::vector<std::string> DeviceWorkspaceManager::getWorkspaceIdsForCurrentThread() {
    std::vector<std::string> ids;
    ThreadWorkspaceEntry* entry = getThreadEntry();

    std::lock_guard<std::mutex> lock(entry->mutex);
    ids.reserve(entry->workspaces.size());
    for (const auto& pair : entry->workspaces) {
        ids.push_back(pair.first);
    }
    return ids;
}

sd::LongType DeviceWorkspaceManager::getTotalMemoryUsage() {
    sd::LongType total = 0;

    std::lock_guard<std::mutex> readLock(_threadWorkspacesMutex);
    for (const auto& threadPair : _threadWorkspaces) {
        std::lock_guard<std::mutex> lock(threadPair.second->mutex);
        for (const auto& wsPair : threadPair.second->workspaces) {
            if (wsPair.second != nullptr) {
                total += wsPair.second->getTotalAllocatedSize();
            }
        }
    }
    return total;
}

std::unordered_map<std::string, sd::LongType> DeviceWorkspaceManager::getMemoryUsagePerDevice() {
    std::unordered_map<std::string, sd::LongType> usage;

    std::lock_guard<std::mutex> readLock(_threadWorkspacesMutex);
    for (const auto& threadPair : _threadWorkspaces) {
        std::lock_guard<std::mutex> lock(threadPair.second->mutex);
        for (const auto& wsPair : threadPair.second->workspaces) {
            if (wsPair.second != nullptr) {
                auto devices = wsPair.second->getActiveDevices();
                for (const auto& device : devices) {
                    sd::LongType deviceUsage = wsPair.second->getAllocatedSizeOnDevice(device);
                    usage[device.deviceId] += deviceUsage;
                }
            }
        }
    }
    return usage;
}

void DeviceWorkspaceManager::printAllocationStatistics() {
    sd_printf("=== Device Workspace Manager Statistics ===\n", "");
    sd_printf("Total workspaces created: %lld\n", _totalWorkspacesCreated.load());
    sd_printf("Total workspaces destroyed: %lld\n", _totalWorkspacesDestroyed.load());
    sd_printf("Total memory currently allocated: %lld bytes\n", getTotalMemoryUsage());

    auto perDevice = getMemoryUsagePerDevice();
    sd_printf("Memory per device:\n", "");
    for (const auto& pair : perDevice) {
        sd_printf("  %s: %lld bytes\n", pair.first.c_str(), pair.second);
    }
}

void DeviceWorkspaceManager::scopeOutOfAllWorkspaces() {
    ThreadWorkspaceEntry* entry = getThreadEntry();
    std::lock_guard<std::mutex> lock(entry->mutex);

    for (auto& pair : entry->workspaces) {
        if (pair.second != nullptr && pair.second->isScopeActive()) {
            pair.second->scopeOut();
        }
    }
}

// ========================
// C-style API Implementation
// ========================

extern "C" {

DeviceWorkspaceManagerHandle getDeviceWorkspaceManager() {
    return DeviceWorkspaceManager::getInstance();
}

void shutdownDeviceWorkspaceManager() {
    DeviceWorkspaceManager::shutdown();
}

MultiBackendWorkspaceHandle dwmCreateWorkspace(
    DeviceWorkspaceManagerHandle manager,
    sd::LongType initialSize,
    const char* id) {

    if (manager == nullptr) return nullptr;
    std::string idStr = (id != nullptr) ? id : "";
    return manager->createWorkspace(initialSize, idStr);
}

MultiBackendWorkspaceHandle dwmCreateWorkspaceWithConfig(
    DeviceWorkspaceManagerHandle manager,
    sd::LongType initialSize,
    sd::LongType maxSize,
    bool crossDeviceMirroring,
    bool asyncTransfers,
    int primaryDeviceType,
    int primaryDeviceIndex,
    const char* id) {

    if (manager == nullptr) return nullptr;

    MultiBackendWorkspaceConfig config;
    config.initialSize = initialSize;
    config.maxSize = maxSize;
    config.crossDeviceMirroring = crossDeviceMirroring;
    config.asyncTransfers = asyncTransfers;
    config.primaryDevice = DeviceDescriptor(static_cast<DeviceType>(primaryDeviceType),
                                             primaryDeviceIndex);

    std::string idStr = (id != nullptr) ? id : "";
    return manager->createWorkspace(config, idStr);
}

MultiBackendWorkspaceHandle dwmGetWorkspace(
    DeviceWorkspaceManagerHandle manager,
    const char* id) {

    if (manager == nullptr || id == nullptr) return nullptr;
    return manager->getWorkspace(id);
}

MultiBackendWorkspaceHandle dwmGetOrCreateWorkspace(
    DeviceWorkspaceManagerHandle manager,
    sd::LongType initialSize,
    const char* id) {

    if (manager == nullptr || id == nullptr) return nullptr;

    MultiBackendWorkspaceConfig config = manager->getDefaultConfiguration();
    config.initialSize = initialSize;
    return manager->getOrCreateWorkspace(config, id);
}

MultiBackendWorkspaceHandle dwmGetAndActivateWorkspace(
    DeviceWorkspaceManagerHandle manager,
    sd::LongType initialSize,
    const char* id) {

    if (manager == nullptr || id == nullptr) return nullptr;

    MultiBackendWorkspaceConfig config = manager->getDefaultConfiguration();
    config.initialSize = initialSize;
    return manager->getAndActivateWorkspace(config, id);
}

bool dwmWorkspaceExists(
    DeviceWorkspaceManagerHandle manager,
    const char* id) {

    if (manager == nullptr || id == nullptr) return false;
    return manager->workspaceExists(id);
}

void dwmDestroyWorkspace(
    DeviceWorkspaceManagerHandle manager,
    MultiBackendWorkspaceHandle workspace) {

    if (manager != nullptr && workspace != nullptr) {
        manager->destroyWorkspace(workspace);
    }
}

void dwmDestroyWorkspaceById(
    DeviceWorkspaceManagerHandle manager,
    const char* id) {

    if (manager != nullptr && id != nullptr) {
        manager->destroyWorkspace(id);
    }
}

void dwmDestroyAllWorkspacesForCurrentThread(DeviceWorkspaceManagerHandle manager) {
    if (manager != nullptr) {
        manager->destroyAllWorkspacesForCurrentThread();
    }
}

void dwmMakeGlobal(
    DeviceWorkspaceManagerHandle manager,
    MultiBackendWorkspaceHandle workspace) {

    if (manager != nullptr && workspace != nullptr) {
        manager->makeGlobal(workspace);
    }
}

MultiBackendWorkspaceHandle dwmGetGlobalWorkspace(
    DeviceWorkspaceManagerHandle manager,
    const char* id) {

    if (manager == nullptr || id == nullptr) return nullptr;
    return manager->getGlobalWorkspace(id);
}

sd::LongType dwmGetTotalMemoryUsage(DeviceWorkspaceManagerHandle manager) {
    if (manager == nullptr) return 0;
    return manager->getTotalMemoryUsage();
}

int dwmGetWorkspaceCountForCurrentThread(DeviceWorkspaceManagerHandle manager) {
    if (manager == nullptr) return 0;
    return static_cast<int>(manager->getWorkspaceIdsForCurrentThread().size());
}

void dwmScopeOutOfAllWorkspaces(DeviceWorkspaceManagerHandle manager) {
    if (manager != nullptr) {
        manager->scopeOutOfAllWorkspaces();
    }
}

}  // extern "C"

}  // namespace memory
}  // namespace sd
