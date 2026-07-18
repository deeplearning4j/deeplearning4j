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

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>

#include <algorithm>
#include <cstring>
#include <utility>

// ---------------------------------------------------------------------------
// Thread-local current device  (matches cudaSetDevice / AffinityManager)
// ---------------------------------------------------------------------------
namespace {
thread_local int tl_currentVulkanDevice = 0;
}

namespace sd {
namespace graph {

// ── Singleton ──────────────────────────────────────────────────────────────

VulkanDeviceManager& VulkanDeviceManager::getInstance() {
  // Keep the singleton object itself alive for the process, matching CUDA's
  // LaunchContext ownership pattern. Vulkan additionally owns ICD worker threads,
  // so VulkanBackend invokes shutdown() explicitly while the loader is live;
  // retaining the C++ wrapper prevents a second DSO-destruction pass afterward.
  static VulkanDeviceManager* instance = new VulkanDeviceManager();
  return *instance;
}

VulkanDeviceManager::VulkanDeviceManager() {}

VulkanDeviceManager::~VulkanDeviceManager() { shutdown(); }

void VulkanDeviceManager::shutdown() {
  std::lock_guard<std::mutex> lk(mutex_);
  if (instance_ == VK_NULL_HANDLE) return;
  if (VulkanDeviceContext::hasLiveContexts()) {
    sd_printf("VulkanDeviceManager::shutdown: refusing to destroy VkInstance "
              "while logical-device contexts are still live\n");
    return;
  }

  // The backend lifecycle drains VulkanMemoryPool and destroys every
  // VulkanDeviceContext before the shared instance reaches this final phase.
  // This manager deliberately owns no VkDevice objects.
  vkDestroyInstance(instance_, nullptr);
  instance_ = VK_NULL_HANDLE;

  devices_.clear();
  trackedBytes_.reset();
  trackedBytesCount_ = 0;
  negotiatedApiVersion_ = 0;
  physicalDeviceFeatures2Fn_ = nullptr;
  physicalDeviceMemoryProperties2Fn_ = nullptr;
  initialized_ = false;
  initFailed_ = false;
}

// ── Initialization ─────────────────────────────────────────────────────────

bool VulkanDeviceManager::initialize() {
  std::lock_guard<std::mutex> lk(mutex_);
  if (initialized_) return true;
  if (initFailed_) return false;
  bool ok = doInitialize();
  if (ok) {
    initialized_ = true;
  } else {
    initFailed_ = true;
  }
  return ok;
}

bool VulkanDeviceManager::isInitialized() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return initialized_;
}

bool VulkanDeviceManager::doInitialize() {
  // ── Create VkInstance with loader-supported API version ──────────────
  // Vulkan 1.0 loaders (including Android API 24 devices) do not export
  // vkEnumerateInstanceVersion. In that case the loader version is 1.0.
  uint32_t loaderApiVersion = VK_API_VERSION_1_0;
  auto enumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
      vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
  if (enumerateInstanceVersion != nullptr) {
    VkResult versionResult = enumerateInstanceVersion(&loaderApiVersion);
    if (versionResult != VK_SUCCESS) loaderApiVersion = VK_API_VERSION_1_0;
  }
  const uint32_t negotiatedApiVersion =
      std::min(loaderApiVersion, VK_API_VERSION_1_2);

  // Vulkan 1.0 exposes Features2/Properties2 through an instance extension.
  // Enable it when advertised so Android API-24 and other 1.0 stacks can use
  // device extensions whose feature/property structures depend on that path.
  std::vector<const char*> enabledInstanceExtensions;
  bool properties2ExtensionEnabled = false;
  if (negotiatedApiVersion < VK_API_VERSION_1_1) {
    std::vector<VkExtensionProperties> instanceExtensions;
    VkResult extensionResult = VK_SUCCESS;
    do {
      uint32_t extensionCount = 0;
      extensionResult = vkEnumerateInstanceExtensionProperties(
          nullptr, &extensionCount, nullptr);
      if (extensionResult != VK_SUCCESS) {
        sd_printf("VulkanDeviceManager: instance-extension count failed "
                  "(result=%d)\n",
                  static_cast<int>(extensionResult));
        return false;
      }
      instanceExtensions.resize(extensionCount);
      extensionResult = vkEnumerateInstanceExtensionProperties(
          nullptr, &extensionCount,
          instanceExtensions.empty() ? nullptr : instanceExtensions.data());
      instanceExtensions.resize(extensionCount);
    } while (extensionResult == VK_INCOMPLETE);
    if (extensionResult != VK_SUCCESS) {
      sd_printf("VulkanDeviceManager: instance-extension enumeration failed "
                "(result=%d)\n",
                static_cast<int>(extensionResult));
      return false;
    }

    properties2ExtensionEnabled = std::any_of(
        instanceExtensions.begin(), instanceExtensions.end(),
        [](const VkExtensionProperties& extension) {
          return strcmp(
                     extension.extensionName,
                     VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) ==
                 0;
        });
    if (properties2ExtensionEnabled) {
      enabledInstanceExtensions.push_back(
          VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
  }

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "nd4j-vulkan-devmgr";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "libnd4j";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = negotiatedApiVersion;

  VkInstanceCreateInfo instInfo = {};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pApplicationInfo = &appInfo;
  instInfo.enabledExtensionCount =
      static_cast<uint32_t>(enabledInstanceExtensions.size());
  instInfo.ppEnabledExtensionNames =
      enabledInstanceExtensions.empty()
          ? nullptr
          : enabledInstanceExtensions.data();

  VkResult res = vkCreateInstance(&instInfo, nullptr, &instance_);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanDeviceManager: vkCreateInstance failed for apiVersion=%u.%u (result=%d)\n",
              static_cast<unsigned>(VK_VERSION_MAJOR(negotiatedApiVersion)),
              static_cast<unsigned>(VK_VERSION_MINOR(negotiatedApiVersion)),
              static_cast<int>(res));
    return false;
  }

  const bool coreProperties2 =
      negotiatedApiVersion >= VK_API_VERSION_1_1;
  if (coreProperties2 || properties2ExtensionEnabled) {
    physicalDeviceFeatures2Fn_ =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            vkGetInstanceProcAddr(
                instance_, coreProperties2
                               ? "vkGetPhysicalDeviceFeatures2"
                               : "vkGetPhysicalDeviceFeatures2KHR"));
    physicalDeviceMemoryProperties2Fn_ =
        reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
            vkGetInstanceProcAddr(
                instance_, coreProperties2
                               ? "vkGetPhysicalDeviceMemoryProperties2"
                               : "vkGetPhysicalDeviceMemoryProperties2KHR"));
    if (physicalDeviceFeatures2Fn_ == nullptr ||
        physicalDeviceMemoryProperties2Fn_ == nullptr) {
      sd_printf("VulkanDeviceManager: negotiated %s but required "
                "Features2/MemoryProperties2 entry points are unavailable\n",
                coreProperties2
                    ? "Vulkan 1.1 core"
                    : "VK_KHR_get_physical_device_properties2");
      vkDestroyInstance(instance_, nullptr);
      instance_ = VK_NULL_HANDLE;
      physicalDeviceFeatures2Fn_ = nullptr;
      physicalDeviceMemoryProperties2Fn_ = nullptr;
      return false;
    }
  }
  negotiatedApiVersion_ = negotiatedApiVersion;

  // ── Enumerate physical devices ──────────────────────────────────────
  std::vector<VkPhysicalDevice> physDevs;
  do {
    uint32_t physCount = 0;
    res = vkEnumeratePhysicalDevices(instance_, &physCount, nullptr);
    if (res != VK_SUCCESS || physCount == 0) {
      sd_printf("VulkanDeviceManager: physical-device count failed or "
                "returned no devices (result=%d)\n",
                static_cast<int>(res));
      vkDestroyInstance(instance_, nullptr);
      instance_ = VK_NULL_HANDLE;
      physicalDeviceFeatures2Fn_ = nullptr;
      physicalDeviceMemoryProperties2Fn_ = nullptr;
      return false;
    }

    physDevs.resize(physCount);
    res = vkEnumeratePhysicalDevices(instance_, &physCount, physDevs.data());
    physDevs.resize(physCount);
  } while (res == VK_INCOMPLETE);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanDeviceManager: physical-device enumeration failed "
              "(result=%d)\n",
              static_cast<int>(res));
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    physicalDeviceFeatures2Fn_ = nullptr;
    physicalDeviceMemoryProperties2Fn_ = nullptr;
    return false;
  }
  const uint32_t physCount = static_cast<uint32_t>(physDevs.size());

  // Sort: discrete GPUs first, then by largest DEVICE_LOCAL heap (descending).
  // This matches the convention that device 0 is the "best" device, consistent
  // with CUDA's default ordering heuristic.
  auto deviceTypePriority = [](VkPhysicalDeviceType t) -> int {
    switch (t) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 0;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 3;
      default:                                      return 4;
    }
  };

  struct SortEntry {
    VkPhysicalDevice phys;
    int typePriority;
    uint64_t largestHeap;
    uint32_t computeQueueFamily;
  };
  std::vector<SortEntry> sortBuf;
  sortBuf.reserve(physCount);

  for (auto& phys : physDevs) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    uint64_t largest = 0;
    for (uint32_t hi = 0; hi < memProps.memoryHeapCount; ++hi) {
      if (memProps.memoryHeaps[hi].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        if (memProps.memoryHeaps[hi].size > largest)
          largest = memProps.memoryHeaps[hi].size;
      }
    }

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(qfCount);
    if (qfCount > 0) {
      vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, queueFamilies.data());
    }

    bool hasComputeQueue = false;
    uint32_t computeQueueFamily = 0;
    for (uint32_t qf = 0; qf < qfCount; ++qf) {
      if (queueFamilies[qf].queueCount > 0 &&
          (queueFamilies[qf].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        computeQueueFamily = qf;
        hasComputeQueue = true;
        break;
      }
    }
    if (!hasComputeQueue) {
      sd_printf("VulkanDeviceManager: physical device \"%s\" has no compute queue; excluding it\n",
                props.deviceName);
      continue;
    }

    sortBuf.push_back(
        {phys, deviceTypePriority(props.deviceType), largest, computeQueueFamily});
  }

  std::stable_sort(sortBuf.begin(), sortBuf.end(), [](const SortEntry& a, const SortEntry& b) {
    if (a.typePriority != b.typePriority) return a.typePriority < b.typePriority;
    return a.largestHeap > b.largestHeap;
  });

  // ── Build device info for compute-capable physical adapters ──────────
  // Logical devices are created exactly once by VulkanDeviceContext. Keeping
  // enumeration here vendor-neutral lets the installed ICD supply AMD, NVIDIA,
  // Intel, Android, or software implementations without backend-specific paths.
  devices_.clear();
  devices_.reserve(sortBuf.size());

  for (const auto& entry : sortBuf) {
    VkPhysicalDevice phys = entry.phys;
    VulkanDeviceInfo info;
    info.physicalDevice = phys;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    info.name = props.deviceName;
    info.apiVersionRaw = props.apiVersion;
    info.vendorId = props.vendorID;
    info.vkMajor = static_cast<int>(VK_VERSION_MAJOR(props.apiVersion));
    info.vkMinor = static_cast<int>(VK_VERSION_MINOR(props.apiVersion));
    info.totalMemoryBytes = entry.largestHeap;

    // Check for VK_EXT_memory_budget device extension.
    std::vector<VkExtensionProperties> devExts;
    VkResult devExtResult = VK_SUCCESS;
    do {
      uint32_t devExtCount = 0;
      devExtResult = vkEnumerateDeviceExtensionProperties(
          phys, nullptr, &devExtCount, nullptr);
      if (devExtResult != VK_SUCCESS) break;
      devExts.resize(devExtCount);
      devExtResult = vkEnumerateDeviceExtensionProperties(
          phys, nullptr, &devExtCount,
          devExts.empty() ? nullptr : devExts.data());
      devExts.resize(devExtCount);
    } while (devExtResult == VK_INCOMPLETE);
    if (devExtResult != VK_SUCCESS) {
      sd_printf("VulkanDeviceManager: device-extension enumeration failed "
                "for \"%s\" (result=%d); excluding device\n",
                info.name.c_str(), static_cast<int>(devExtResult));
      continue;
    }
    for (auto& ext : devExts) {
      if (strcmp(ext.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) ==
          0) {
        info.memoryBudgetExt =
            physicalDeviceMemoryProperties2Fn_ != nullptr;
        break;
      }
    }

    const size_t deviceIndex = devices_.size();
    DSP_DIAG(BACKEND,
             "VulkanDeviceManager: device[%zu] \"%s\" vendor=0x%04x apiVersion=%d.%d "
             "totalMem=%lluMB computeQueue=%u budget_ext=%s",
             deviceIndex, info.name.c_str(), static_cast<unsigned>(info.vendorId),
             info.vkMajor, info.vkMinor,
             (unsigned long long)(info.totalMemoryBytes / (1024 * 1024)),
             static_cast<unsigned>(entry.computeQueueFamily),
             info.memoryBudgetExt ? "yes" : "no");
    devices_.push_back(std::move(info));
  }

  if (devices_.empty()) {
    sd_printf("VulkanDeviceManager: no compute-capable Vulkan physical device was enumerated\n");
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    negotiatedApiVersion_ = 0;
    physicalDeviceFeatures2Fn_ = nullptr;
    physicalDeviceMemoryProperties2Fn_ = nullptr;
    return false;
  }

  // std::atomic is non-moveable, so use a heap array instead of vector.
  trackedBytesCount_ = devices_.size();
  trackedBytes_ = std::unique_ptr<std::atomic<uint64_t>[]>(
      new std::atomic<uint64_t>[trackedBytesCount_]);
  for (size_t i = 0; i < trackedBytesCount_; ++i) trackedBytes_[i].store(0);

  DSP_DIAG(BACKEND, "VulkanDeviceManager: initialized with %zu device(s)", devices_.size());
  return true;
}

// ── Device enumeration accessors ─────────────────────────────────────────

int VulkanDeviceManager::deviceCount() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return static_cast<int>(devices_.size());
}

VkPhysicalDevice VulkanDeviceManager::getPhysicalDevice(int deviceId) const {
  std::lock_guard<std::mutex> lk(mutex_);
  if (deviceId < 0 || deviceId >= static_cast<int>(devices_.size()))
    return VK_NULL_HANDLE;
  return devices_[static_cast<size_t>(deviceId)].physicalDevice;
}

const VulkanDeviceInfo* VulkanDeviceManager::getDeviceInfo(int deviceId) const {
  std::lock_guard<std::mutex> lk(mutex_);
  if (deviceId < 0 || deviceId >= static_cast<int>(devices_.size()))
    return nullptr;
  return &devices_[static_cast<size_t>(deviceId)];
}

// ── Thread-local current device ───────────────────────────────────────────

int VulkanDeviceManager::currentDeviceId() {
  return tl_currentVulkanDevice;
}

bool VulkanDeviceManager::setCurrentDevice(int deviceId) {
  // Validate against device count only if already initialized.
  if (initialized_) {
    if (deviceId < 0 || deviceId >= static_cast<int>(devices_.size()))
      return false;
  }
  tl_currentVulkanDevice = deviceId;
  return true;
}

// ── Memory budget tracking ────────────────────────────────────────────────

void VulkanDeviceManager::trackAllocation(int deviceId, size_t bytes) {
  if (!trackedBytes_ || deviceId < 0 || static_cast<size_t>(deviceId) >= trackedBytesCount_) return;
  trackedBytes_[static_cast<size_t>(deviceId)].fetch_add(static_cast<uint64_t>(bytes),
                                                          std::memory_order_relaxed);
}

void VulkanDeviceManager::untrackAllocation(int deviceId, size_t bytes) {
  if (!trackedBytes_ || deviceId < 0 || static_cast<size_t>(deviceId) >= trackedBytesCount_) return;
  uint64_t cur = trackedBytes_[static_cast<size_t>(deviceId)].load(std::memory_order_relaxed);
  uint64_t sub = static_cast<uint64_t>(bytes);
  // Clamp to 0 — atomic compare-exchange loop for correctness.
  uint64_t desired;
  do {
    desired = (cur >= sub) ? (cur - sub) : 0u;
  } while (!trackedBytes_[static_cast<size_t>(deviceId)].compare_exchange_weak(
      cur, desired, std::memory_order_relaxed, std::memory_order_relaxed));
}

uint64_t VulkanDeviceManager::getFreeMemory(int deviceId) const {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!initialized_ || deviceId < 0 || deviceId >= static_cast<int>(devices_.size()))
    return 0;

  const VulkanDeviceInfo& info = devices_[static_cast<size_t>(deviceId)];

  // Prefer VK_EXT_memory_budget for an accurate driver-reported budget.
  if (info.memoryBudgetExt &&
      physicalDeviceMemoryProperties2Fn_ != nullptr) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps = {};
    budgetProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    budgetProps.pNext = nullptr;

    VkPhysicalDeviceMemoryProperties2 memProps2 = {};
    memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memProps2.pNext = &budgetProps;

    physicalDeviceMemoryProperties2Fn_(info.physicalDevice, &memProps2);

    // Sum available budget across DEVICE_LOCAL heaps.
    const VkPhysicalDeviceMemoryProperties& mp = memProps2.memoryProperties;
    uint64_t freeBytes = 0;
    for (uint32_t hi = 0; hi < mp.memoryHeapCount; ++hi) {
      if (mp.memoryHeaps[hi].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        VkDeviceSize budget = budgetProps.heapBudget[hi];
        VkDeviceSize usage  = budgetProps.heapUsage[hi];
        if (budget > usage) freeBytes += (budget - usage);
      }
    }
    return freeBytes;
  }

  // Fallback: total_device_local_mem - tracked_allocations.
  uint64_t tracked = (trackedBytes_ && static_cast<size_t>(deviceId) < trackedBytesCount_)
      ? trackedBytes_[static_cast<size_t>(deviceId)].load(std::memory_order_relaxed) : 0u;
  uint64_t total = info.totalMemoryBytes;
  return (total > tracked) ? (total - tracked) : 0u;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
