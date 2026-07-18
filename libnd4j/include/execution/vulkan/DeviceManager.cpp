/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <execution/DeviceManager.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace sd {
namespace modelparallel {

namespace {

bool validDeviceIndex(const std::vector<DeviceInfo>& devices, int index) {
  return index >= 0 && index < static_cast<int>(devices.size());
}

bool directVulkanAccess(int source, int target) {
  return graph::VulkanExecutionStream::isCrossDeviceCopySupported(source, target) &&
         graph::VulkanExecutionStream::isCrossDeviceCopySupported(target, source);
}

std::string vendorIdString(uint32_t vendorId) {
  std::ostringstream stream;
  stream << "Vulkan vendor 0x" << std::hex << std::setw(4) << std::setfill('0')
         << vendorId;
  return stream.str();
}

}  // namespace

int& DeviceManager::currentDeviceRef() {
  static thread_local int currentDevice = 0;
  return currentDevice;
}

DeviceManager& DeviceManager::getInstance() {
  static DeviceManager* instance = new DeviceManager();
  return *instance;
}

DeviceManager::DeviceManager() = default;

DeviceManager::~DeviceManager() { shutdown(); }

bool DeviceManager::initialize() {
  if (_initialized.load(std::memory_order_acquire)) return true;

  auto& vulkan = graph::VulkanDeviceManager::getInstance();
  if (!vulkan.initialize()) return false;

  std::lock_guard<std::mutex> lock(_mutex);
  if (_initialized.load(std::memory_order_relaxed)) return true;

  _devices.clear();
  _devicesByType.clear();
  _p2pConnections.clear();
  _deviceGroups.clear();
  _reservedMemory.clear();
  discoverVulkanDevices();
  initializeP2PConnections();
  const bool ready = !_devices.empty();
  _initialized.store(ready, std::memory_order_release);
  return ready;
}

void DeviceManager::refresh() {
  if (_initialized.load(std::memory_order_acquire)) {
    updateAllMemoryStats();
  } else {
    initialize();
  }
}

void DeviceManager::shutdown() {
  if (!_initialized.exchange(false, std::memory_order_acq_rel)) return;

  graph::VulkanExecutionStream::destroyAll();
  graph::VulkanDeviceContext::destroyAll();

  std::lock_guard<std::mutex> lock(_mutex);
  _devices.clear();
  _devicesByType.clear();
  _p2pConnections.clear();
  _deviceGroups.clear();
  _reservedMemory.clear();
  currentDeviceRef() = 0;
  graph::VulkanDeviceManager::getInstance().shutdown();
}

void DeviceManager::discoverVulkanDevices() {
  auto& vulkan = graph::VulkanDeviceManager::getInstance();
  const int count = vulkan.deviceCount();
  _devices.reserve(static_cast<size_t>(count));

  for (int localIndex = 0; localIndex < count; ++localIndex) {
    const auto* vkInfo = vulkan.getDeviceInfo(localIndex);
    const VkPhysicalDevice physical = vulkan.getPhysicalDevice(localIndex);
    if (vkInfo == nullptr || physical == VK_NULL_HANDLE) continue;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);

    DeviceInfo device;
    device.type = DeviceType::VULKAN_GPU;
    device.deviceIndex = localIndex;
    device.globalIndex = static_cast<int>(_devices.size());
    device.name = vkInfo->name;
    device.vendor = vendorIdString(vkInfo->vendorId);
    device.driverVersion = std::to_string(properties.driverVersion);
    device.totalMemory = static_cast<size_t>(vkInfo->totalMemoryBytes);
    device.freeMemory = static_cast<size_t>(vulkan.getFreeMemory(localIndex));
    device.availableMemory = device.freeMemory;
    device.computeCapabilityMajor = vkInfo->vkMajor;
    device.computeCapabilityMinor = vkInfo->vkMinor;
    device.numCores =
        static_cast<int>(properties.limits.maxComputeWorkGroupInvocations);
    device.numSMs = 0;
    device.clockSpeedGHz = 0.0f;
    device.theoreticalTFlops = 0.0f;
    device.supportsP2P = false;
    device.supportsUnifiedMemory = false;
    device.supportsAsyncCopy = true;
    device.available = true;
    device.engine = samediff::ENGINE_VULKAN;

    _devicesByType[DeviceType::VULKAN_GPU].push_back(device.globalIndex);
    _devices.push_back(std::move(device));
  }
}

void DeviceManager::initializeP2PConnections() {
  for (int source = 0; source < static_cast<int>(_devices.size()); ++source) {
    for (int target = source + 1; target < static_cast<int>(_devices.size());
         ++target) {
      probeP2PCapabilities(source, target);
    }
  }
}

void DeviceManager::probeP2PCapabilities(int device1, int device2) {
  if (!validDeviceIndex(_devices, device1) ||
      !validDeviceIndex(_devices, device2) || device1 == device2 ||
      !directVulkanAccess(_devices[device1].deviceIndex,
                          _devices[device2].deviceIndex)) {
    return;
  }

  P2PConnection connection;
  connection.sourceDevice = device1;
  connection.targetDevice = device2;
  connection.directAccess = true;
  connection.nvlinkConnected = false;
  connection.bidirectional = true;
  _p2pConnections.push_back(connection);
  _devices[device1].supportsP2P = true;
  _devices[device2].supportsP2P = true;
  _devices[device1].p2pConnectedDevices.push_back(device2);
  _devices[device2].p2pConnectedDevices.push_back(device1);
}

int DeviceManager::getTotalDeviceCount() const {
  std::lock_guard<std::mutex> lock(_mutex);
  return static_cast<int>(_devices.size());
}

int DeviceManager::getDeviceCount(DeviceType type) const {
  std::lock_guard<std::mutex> lock(_mutex);
  if (type == DeviceType::ANY) return static_cast<int>(_devices.size());
  const auto it = _devicesByType.find(type);
  return it == _devicesByType.end() ? 0 : static_cast<int>(it->second.size());
}

int DeviceManager::getCpuCount() const { return 0; }

int DeviceManager::getCudaGpuCount() const { return 0; }

std::vector<DeviceInfo> DeviceManager::getAllDevices() const {
  std::lock_guard<std::mutex> lock(_mutex);
  return _devices;
}

std::vector<DeviceInfo> DeviceManager::getDevices(DeviceType type) const {
  std::lock_guard<std::mutex> lock(_mutex);
  if (type == DeviceType::ANY) return _devices;
  std::vector<DeviceInfo> result;
  const auto it = _devicesByType.find(type);
  if (it != _devicesByType.end()) {
    result.reserve(it->second.size());
    for (const int index : it->second) result.push_back(_devices[index]);
  }
  return result;
}

DeviceInfo DeviceManager::getDeviceInfo(int globalIndex) const {
  std::lock_guard<std::mutex> lock(_mutex);
  const auto* device = findDevice(globalIndex);
  return device == nullptr ? DeviceInfo{} : *device;
}

DeviceInfo DeviceManager::getDeviceInfo(DeviceType type, int localIndex) const {
  std::lock_guard<std::mutex> lock(_mutex);
  const int globalIndex = findGlobalIndex(type, localIndex);
  return globalIndex < 0 ? DeviceInfo{} : _devices[globalIndex];
}

bool DeviceManager::isDeviceAvailable(DeviceType type, int localIndex) const {
  std::lock_guard<std::mutex> lock(_mutex);
  const int globalIndex = findGlobalIndex(type, localIndex);
  return globalIndex >= 0 && _devices[globalIndex].available;
}

bool DeviceManager::isDeviceInUse(int globalIndex) const {
  std::lock_guard<std::mutex> lock(_mutex);
  const auto* device = findDevice(globalIndex);
  return device != nullptr && device->inUse;
}

int DeviceManager::findGlobalIndex(DeviceType type, int localIndex) const {
  if (type == DeviceType::ANY) {
    return validDeviceIndex(_devices, localIndex) ? localIndex : -1;
  }
  const auto it = _devicesByType.find(type);
  if (it == _devicesByType.end() || localIndex < 0 ||
      localIndex >= static_cast<int>(it->second.size())) {
    return -1;
  }
  return it->second[localIndex];
}

DeviceInfo* DeviceManager::findDevice(int globalIndex) {
  return validDeviceIndex(_devices, globalIndex) ? &_devices[globalIndex]
                                                 : nullptr;
}

const DeviceInfo* DeviceManager::findDevice(int globalIndex) const {
  return validDeviceIndex(_devices, globalIndex) ? &_devices[globalIndex]
                                                 : nullptr;
}

ParallelCapabilities DeviceManager::getParallelCapabilities() const {
  std::lock_guard<std::mutex> lock(_mutex);
  ParallelCapabilities capabilities;
  const int count = static_cast<int>(_devices.size());
  const size_t requiredP2PConnections =
      count > 1 ? static_cast<size_t>(count * (count - 1) / 2) : 0;
  const bool fullyConnectedP2P =
      count > 1 && _p2pConnections.size() == requiredP2PConnections;

  // The Vulkan collective ABI currently implements only world-size-one
  // operations. Keep every physical device visible, but do not advertise
  // multi-rank tensor/data parallel execution until real collectives exist.
  capabilities.supportsTensorParallel = false;
  capabilities.supportsPipelineParallel = fullyConnectedP2P;
  capabilities.supportsDataParallel = false;
  capabilities.supportsP2P = !_p2pConnections.empty();
  capabilities.supportsAsyncTransfer = count > 0;
  capabilities.maxTensorParallelSize = 1;
  capabilities.maxPipelineStages = fullyConnectedP2P ? count : 1;
  capabilities.maxDataParallelSize = 1;

  bool mixedPrecision = count > 0;
  bool quantization = count > 0;
  for (const auto& device : _devices) {
    auto* context = graph::VulkanDeviceContext::getContext(device.deviceIndex);
    mixedPrecision =
        mixedPrecision && context != nullptr && context->caps().fp16 &&
        context->caps().storage16;
    quantization =
        quantization && context != nullptr && context->caps().int8;
  }
  capabilities.supportsMixedPrecision = mixedPrecision;
  capabilities.supportsQuantization = quantization;
  if (capabilities.supportsP2P) {
    capabilities.supportedCommPatterns.push_back(
        CommunicationPattern::POINT_TO_POINT);
  }
  if (count > 0) {
    capabilities.supportedDeviceTypes.push_back(DeviceType::VULKAN_GPU);
  }
  return capabilities;
}

bool DeviceManager::supportsTensorParallel() const {
  return getParallelCapabilities().supportsTensorParallel;
}

bool DeviceManager::supportsPipelineParallel() const {
  return getParallelCapabilities().supportsPipelineParallel;
}

bool DeviceManager::supportsP2P(int device1, int device2) const {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!validDeviceIndex(_devices, device1) ||
      !validDeviceIndex(_devices, device2)) {
    return false;
  }
  if (device1 == device2) return true;
  for (const auto& connection : _p2pConnections) {
    if ((connection.sourceDevice == device1 &&
         connection.targetDevice == device2) ||
        (connection.sourceDevice == device2 &&
         connection.targetDevice == device1)) {
      return connection.directAccess;
    }
  }
  return false;
}

P2PConnection DeviceManager::getP2PConnection(int device1, int device2) const {
  std::lock_guard<std::mutex> lock(_mutex);
  if (validDeviceIndex(_devices, device1) && device1 == device2) {
    P2PConnection connection;
    connection.sourceDevice = device1;
    connection.targetDevice = device2;
    connection.directAccess = true;
    return connection;
  }
  for (const auto& connection : _p2pConnections) {
    if ((connection.sourceDevice == device1 &&
         connection.targetDevice == device2) ||
        (connection.sourceDevice == device2 &&
         connection.targetDevice == device1)) {
      return connection;
    }
  }
  return P2PConnection{};
}

std::vector<P2PConnection> DeviceManager::getAllP2PConnections() const {
  std::lock_guard<std::mutex> lock(_mutex);
  return _p2pConnections;
}

bool DeviceManager::enableP2P(int device1, int device2) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!validDeviceIndex(_devices, device1) ||
      !validDeviceIndex(_devices, device2)) {
    return false;
  }
  if (device1 == device2) return true;
  return directVulkanAccess(_devices[device1].deviceIndex,
                            _devices[device2].deviceIndex);
}

void DeviceManager::disableP2P(int device1, int device2) {
  (void)device1;
  (void)device2;
  // Vulkan external-memory capability belongs to the logical devices and cannot
  // be toggled per pair. This method intentionally leaves that capability intact.
}

void DeviceManager::enableAllP2P() {
  std::vector<P2PConnection> connections;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    connections = _p2pConnections;
  }
  for (const auto& connection : connections) {
    enableP2P(connection.sourceDevice, connection.targetDevice);
  }
}

DeviceAllocation DeviceManager::allocateDevices(
    const ModelParallelConfig& config) {
  std::lock_guard<std::mutex> lock(_mutex);
  DeviceAllocation result;
  for (const auto& spec : config.devices) {
    const int index = findGlobalIndex(spec.type, spec.deviceIndex);
    if (index < 0) {
      result.errorMessage = "Device not found: " + deviceTypeToString(spec.type) +
                            " index " + std::to_string(spec.deviceIndex);
      return result;
    }
    if (_devices[index].inUse) {
      result.errorMessage = "Device already in use: " + _devices[index].name;
      return result;
    }
    result.allocatedDevices.push_back(_devices[index]);
  }
  for (const auto& device : result.allocatedDevices) {
    auto& stored = _devices[device.globalIndex];
    stored.inUse = true;
    ++stored.currentUserCount;
  }
  result.success = true;
  return result;
}

DeviceAllocation DeviceManager::allocateDevices(DeviceType type, int count) {
  DeviceAllocation result;
  if (count < 0) {
    result.errorMessage = "Device count must be non-negative";
    return result;
  }

  std::lock_guard<std::mutex> lock(_mutex);
  const auto it = _devicesByType.find(type);
  if (it == _devicesByType.end() ||
      static_cast<int>(it->second.size()) < count) {
    result.errorMessage =
        "Not enough devices of type " + deviceTypeToString(type);
    return result;
  }
  for (const int index : it->second) {
    if (!_devices[index].inUse &&
        static_cast<int>(result.allocatedDevices.size()) < count) {
      result.allocatedDevices.push_back(_devices[index]);
    }
  }
  if (static_cast<int>(result.allocatedDevices.size()) != count) {
    result.allocatedDevices.clear();
    result.errorMessage = "Not enough available devices";
    return result;
  }
  for (const auto& device : result.allocatedDevices) {
    auto& stored = _devices[device.globalIndex];
    stored.inUse = true;
    ++stored.currentUserCount;
  }
  result.success = true;
  return result;
}

DeviceGroup DeviceManager::createDeviceGroup(
    const std::string& name, const std::vector<DeviceSpec>& specs) {
  std::lock_guard<std::mutex> lock(_mutex);
  DeviceGroup group;
  group.name = name;
  for (const auto& spec : specs) {
    const int index = findGlobalIndex(spec.type, spec.deviceIndex);
    if (index < 0) return DeviceGroup{};
    group.deviceIndices.push_back(index);
  }
  group.initialized = true;
  _deviceGroups[name] = group;
  return group;
}

void DeviceManager::releaseDeviceGroup(const std::string& name) {
  std::lock_guard<std::mutex> lock(_mutex);
  _deviceGroups.erase(name);
}

DeviceGroup DeviceManager::getDeviceGroup(const std::string& name) const {
  std::lock_guard<std::mutex> lock(_mutex);
  const auto it = _deviceGroups.find(name);
  return it == _deviceGroups.end() ? DeviceGroup{} : it->second;
}

void DeviceManager::acquireDevice(int globalIndex) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (auto* device = findDevice(globalIndex)) {
    device->inUse = true;
    ++device->currentUserCount;
  }
}

void DeviceManager::releaseDevice(int globalIndex) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (auto* device = findDevice(globalIndex)) {
    device->currentUserCount = std::max(0, device->currentUserCount - 1);
    device->inUse = device->currentUserCount != 0;
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
  const auto it = _devicesByType.find(type);
  if (it != _devicesByType.end()) {
    for (const int index : it->second) total += _devices[index].totalMemory;
  }
  return total;
}

size_t DeviceManager::getFreeMemory(DeviceType type) const {
  std::lock_guard<std::mutex> lock(_mutex);
  size_t total = 0;
  const auto it = _devicesByType.find(type);
  if (it != _devicesByType.end()) {
    for (const int index : it->second) {
      total += static_cast<size_t>(
          graph::VulkanDeviceManager::getInstance().getFreeMemory(
              _devices[index].deviceIndex));
    }
  }
  return total;
}

size_t DeviceManager::getDeviceFreeMemory(int globalIndex) const {
  std::lock_guard<std::mutex> lock(_mutex);
  const auto* device = findDevice(globalIndex);
  return device == nullptr
             ? 0
             : static_cast<size_t>(
                   graph::VulkanDeviceManager::getInstance().getFreeMemory(
                       device->deviceIndex));
}

void DeviceManager::updateMemoryStats(int globalIndex) {
  auto* device = findDevice(globalIndex);
  if (device == nullptr) return;
  device->freeMemory = static_cast<size_t>(
      graph::VulkanDeviceManager::getInstance().getFreeMemory(
          device->deviceIndex));
  const auto reservation = _reservedMemory.find(globalIndex);
  const size_t reserved =
      reservation == _reservedMemory.end() ? 0 : reservation->second;
  device->availableMemory =
      device->freeMemory > reserved ? device->freeMemory - reserved : 0;
}

void DeviceManager::updateAllMemoryStats() {
  std::lock_guard<std::mutex> lock(_mutex);
  for (int index = 0; index < static_cast<int>(_devices.size()); ++index) {
    updateMemoryStats(index);
  }
}

bool DeviceManager::reserveMemory(int globalIndex, size_t bytes) {
  std::lock_guard<std::mutex> lock(_mutex);
  auto* device = findDevice(globalIndex);
  if (device == nullptr || !device->available) return false;
  updateMemoryStats(globalIndex);
  const size_t reserved = _reservedMemory[globalIndex];
  if (bytes > device->freeMemory ||
      reserved > device->freeMemory - bytes) {
    return false;
  }
  _reservedMemory[globalIndex] = reserved + bytes;
  device->availableMemory = device->freeMemory - reserved - bytes;
  return true;
}

void DeviceManager::releaseReservedMemory(int globalIndex, size_t bytes) {
  std::lock_guard<std::mutex> lock(_mutex);
  auto* device = findDevice(globalIndex);
  if (device == nullptr) return;
  auto it = _reservedMemory.find(globalIndex);
  if (it == _reservedMemory.end() || bytes >= it->second) {
    _reservedMemory.erase(globalIndex);
  } else {
    it->second -= bytes;
  }
  updateMemoryStats(globalIndex);
}

std::vector<DeviceInfo> DeviceManager::findBestDevices(
    size_t requiredMemory, int count, bool preferP2P) {
  if (!_initialized.load(std::memory_order_acquire)) initialize();
  if (count <= 0) return {};

  std::lock_guard<std::mutex> lock(_mutex);
  for (int index = 0; index < static_cast<int>(_devices.size()); ++index) {
    updateMemoryStats(index);
  }
  std::vector<DeviceInfo> candidates;
  for (const auto& device : _devices) {
    if (device.available && !device.inUse &&
        device.availableMemory >= requiredMemory) {
      candidates.push_back(device);
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [preferP2P](const DeviceInfo& left, const DeviceInfo& right) {
              if (preferP2P && left.supportsP2P != right.supportsP2P) {
                return left.supportsP2P > right.supportsP2P;
              }
              if (left.availableMemory != right.availableMemory) {
                return left.availableMemory > right.availableMemory;
              }
              return left.globalIndex < right.globalIndex;
            });
  if (count < static_cast<int>(candidates.size())) {
    candidates.resize(static_cast<size_t>(count));
  }
  return candidates;
}

DeviceInfo DeviceManager::getBestGpu() const {
  auto* self = const_cast<DeviceManager*>(this);
  if (!self->_initialized.load(std::memory_order_acquire)) self->initialize();

  std::lock_guard<std::mutex> lock(_mutex);
  for (int index = 0; index < static_cast<int>(_devices.size()); ++index) {
    self->updateMemoryStats(index);
  }
  const DeviceInfo* best = nullptr;
  for (const auto& device : _devices) {
    if (device.type != DeviceType::VULKAN_GPU || !device.available ||
        device.inUse) {
      continue;
    }
    if (best == nullptr || device.availableMemory > best->availableMemory) {
      best = &device;
    }
  }
  return best == nullptr ? DeviceInfo{} : *best;
}

void DeviceManager::setCurrentDevice(int globalIndex) {
  std::lock_guard<std::mutex> lock(_mutex);
  const auto* device = findDevice(globalIndex);
  if (device == nullptr ||
      !graph::VulkanDeviceManager::getInstance().setCurrentDevice(
          device->deviceIndex)) {
    return;
  }
  currentDeviceRef() = globalIndex;
}

void DeviceManager::setCurrentDeviceCuda(const DeviceInfo& device) {
  graph::VulkanDeviceManager::getInstance().setCurrentDevice(
      device.deviceIndex);
}

int DeviceManager::getCurrentDevice() const {
  const int current = currentDeviceRef();
  std::lock_guard<std::mutex> lock(_mutex);
  return validDeviceIndex(_devices, current) ? current : 0;
}

samediff::Engine DeviceManager::deviceTypeToEngine(DeviceType type) {
  switch (type) {
    case DeviceType::VULKAN_GPU:
      return samediff::ENGINE_VULKAN;
    case DeviceType::METAL_GPU:
      return samediff::ENGINE_METAL;
    case DeviceType::OPENCL_GPU:
      return samediff::ENGINE_OPENCL;
    case DeviceType::TPU:
      return samediff::ENGINE_TPU;
    case DeviceType::CUDA_GPU:
      return samediff::ENGINE_CUDA;
    case DeviceType::CPU:
    case DeviceType::ACCELERATOR:
    case DeviceType::ANY:
    default:
      return samediff::ENGINE_CPU;
  }
}

DeviceType DeviceManager::engineToDeviceType(samediff::Engine engine) {
  switch (engine) {
    case samediff::ENGINE_VULKAN:
      return DeviceType::VULKAN_GPU;
    case samediff::ENGINE_METAL:
    case samediff::ENGINE_MPS:
      return DeviceType::METAL_GPU;
    case samediff::ENGINE_OPENCL:
      return DeviceType::OPENCL_GPU;
    case samediff::ENGINE_TPU:
      return DeviceType::TPU;
    case samediff::ENGINE_CUDA:
    case samediff::ENGINE_ZLUDA_AMD:
    case samediff::ENGINE_ZLUDA_INTEL:
      return DeviceType::CUDA_GPU;
    case samediff::ENGINE_ANY:
      return DeviceType::ANY;
    case samediff::ENGINE_CPU:
    case samediff::ENGINE_ACCELERATE:
    case samediff::ENGINE_ONEDNN:
    case samediff::ENGINE_ARM:
    default:
      return DeviceType::CPU;
  }
}

int DeviceManager::getGlobalIndex(samediff::Engine engine,
                                  int localIndex) const {
  std::lock_guard<std::mutex> lock(_mutex);
  return findGlobalIndex(engineToDeviceType(engine), localIndex);
}

void DeviceManager::synchronizeAll() {
  std::vector<int> devices;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto& device : _devices) {
      devices.push_back(device.deviceIndex);
    }
  }
  for (const int device : devices) {
    graph::VulkanExecutionStream::synchronizeDevice(device);
  }
}

void DeviceManager::synchronize(int globalIndex) {
  std::lock_guard<std::mutex> lock(_mutex);
  const auto* device = findDevice(globalIndex);
  if (device != nullptr) {
    graph::VulkanExecutionStream::synchronizeDevice(device->deviceIndex);
  }
}

std::string DeviceManager::deviceToString(const DeviceInfo& device) const {
  std::ostringstream stream;
  stream << "[" << device.globalIndex << "] "
         << deviceTypeToString(device.type) << ":" << device.deviceIndex
         << " \"" << device.name << "\" "
         << (device.totalMemory / (1024 * 1024)) << "MB";
  if (device.inUse) stream << " (in use)";
  return stream.str();
}

void DeviceManager::printDeviceInfo() const {
  std::lock_guard<std::mutex> lock(_mutex);
  std::cout << "=== Vulkan Device Manager ===\n";
  for (const auto& device : _devices) {
    std::cout << "  " << deviceToString(device) << "\n";
  }
}

DeviceContextGuard::DeviceContextGuard(int globalIndex)
    : _deviceIndex(globalIndex),
      _previousDevice(DeviceManager::getInstance().getCurrentDevice()) {
  DeviceManager::getInstance().setCurrentDevice(globalIndex);
}

DeviceContextGuard::DeviceContextGuard(DeviceType type, int localIndex)
    : _deviceIndex(-1),
      _previousDevice(DeviceManager::getInstance().getCurrentDevice()) {
  auto& manager = DeviceManager::getInstance();
  _deviceIndex =
      manager.getGlobalIndex(DeviceManager::deviceTypeToEngine(type), localIndex);
  if (_deviceIndex >= 0) manager.setCurrentDevice(_deviceIndex);
}

DeviceContextGuard::~DeviceContextGuard() {
  DeviceManager::getInstance().setCurrentDevice(_previousDevice);
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
