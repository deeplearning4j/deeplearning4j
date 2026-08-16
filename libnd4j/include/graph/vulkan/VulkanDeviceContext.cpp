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

#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <graph/vulkan/VulkanSpirvDiskCache.h>
#include <graph/vulkan/VulkanPipelineCache.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <system/config/VulkanConfig.h>
#include <helpers/logger.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace sd {
namespace graph {

// ── Static registry ────────────────────────────────────────────────────────

std::mutex VulkanDeviceContext::contextsMtx_;
std::vector<std::unique_ptr<VulkanDeviceContext>> VulkanDeviceContext::contexts_;

// ── Tier-2 pipeline-cache blob persistence helpers (ADR 0115) ──────────────

namespace {

// Spec-defined VK_PIPELINE_CACHE_HEADER_VERSION_ONE prefix of the blob:
// four uint32 fields + the 16-byte pipelineCacheUUID = 32 bytes.
struct PipelineCacheBlobHeader {
  uint32_t headerSize;
  uint32_t headerVersion;
  uint32_t vendorID;
  uint32_t deviceID;
  uint8_t pipelineCacheUUID[VK_UUID_SIZE];
};

// One blob per physical device: vendorID/deviceID/driverVersion/UUID are all
// in the filename hash, so a driver update orphans the old blob instead of
// feeding a stale one back to the driver. Blobs are never shared across
// devices (the JitSegmentCacheKey deviceId lesson, enforced by construction).
std::string pipelineCacheBlobPath(VkPhysicalDevice physicalDevice) {
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physicalDevice, &props);

  static constexpr const char* VULKAN_PIPELINE_BLOB_ABI = "vulkan-driver-pipeline-cache-v1";
  uint64_t hash = dsp::FNV1A64_OFFSET_BASIS;
  dsp::fnv1aMix(hash, VULKAN_PIPELINE_BLOB_ABI, std::strlen(VULKAN_PIPELINE_BLOB_ABI));
  dsp::fnv1aMix(hash, &props.vendorID, sizeof(props.vendorID));
  dsp::fnv1aMix(hash, &props.deviceID, sizeof(props.deviceID));
  dsp::fnv1aMix(hash, &props.driverVersion, sizeof(props.driverVersion));
  dsp::fnv1aMix(hash, props.pipelineCacheUUID, VK_UUID_SIZE);

  std::ostringstream name;
  name << "vkpc_" << std::hex << std::setw(16) << std::setfill('0') << hash << ".bin";

  const std::string dir = VulkanSpirvDiskCache::configuredOrDefaultDir(
      sd::config::VulkanConfig::getInstance().pipelineCacheDir(), "pipeline_cache");
  return dir + "/" + name.str();
}

// Validate the header against the current device before trusting
// pInitialData. The spec obliges drivers to tolerate invalid blobs, but do
// not rely on that (Android driver quality).
bool pipelineCacheBlobMatchesDevice(const std::vector<char>& blob,
                                    VkPhysicalDevice physicalDevice) {
  if (blob.size() < sizeof(PipelineCacheBlobHeader)) return false;
  PipelineCacheBlobHeader header{};
  std::memcpy(&header, blob.data(), sizeof(header));
  if (header.headerSize < sizeof(PipelineCacheBlobHeader)) return false;
  if (header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) return false;
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  if (header.vendorID != props.vendorID || header.deviceID != props.deviceID) return false;
  return std::memcmp(header.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

std::vector<char> readPipelineCacheBlob(VkPhysicalDevice physicalDevice) {
  auto& cfg = sd::config::VulkanConfig::getInstance();
  const std::string path = pipelineCacheBlobPath(physicalDevice);
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) return {};
  std::vector<char> blob((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  const int64_t maxBytes = cfg.pipelineCacheMaxBytes();
  if (maxBytes > 0 && static_cast<int64_t>(blob.size()) > maxBytes) {
    // Growth control: VkPipelineCache has no eviction API — an over-budget
    // blob is dropped and the cache regenerates from scratch.
    DSP_DIAG(BACKEND,
             "VulkanDeviceContext: pipeline-cache blob %s exceeds budget (%zu > %lld) — starting fresh",
             path.c_str(), blob.size(), static_cast<long long>(maxBytes));
    return {};
  }
  if (!pipelineCacheBlobMatchesDevice(blob, physicalDevice)) {
    DSP_DIAG(BACKEND, "VulkanDeviceContext: pipeline-cache blob %s rejected (header/device mismatch)",
             path.c_str());
    return {};
  }
  return blob;
}

}  // namespace

// ── getContext ─────────────────────────────────────────────────────────────

VulkanDeviceContext* VulkanDeviceContext::getContext(int deviceId) {
  VulkanDeviceManager& mgr = VulkanDeviceManager::getInstance();
  if (!mgr.isInitialized() || deviceId < 0 || deviceId >= mgr.deviceCount())
    return nullptr;

  std::lock_guard<std::mutex> lk(contextsMtx_);
  size_t idx = static_cast<size_t>(deviceId);
  if (idx >= contexts_.size()) contexts_.resize(idx + 1);
  if (!contexts_[idx]) {
    auto ctx = std::unique_ptr<VulkanDeviceContext>(new VulkanDeviceContext(deviceId));
    if (!ctx->initialize()) {
      sd_printf("VulkanDeviceContext: failed to initialize context for device %d\n", deviceId);
      return nullptr;
    }
    contexts_[idx] = std::move(ctx);
  }
  return contexts_[idx].get();
}

// ── destroyAll ─────────────────────────────────────────────────────────────

void VulkanDeviceContext::destroyAll() {
  std::lock_guard<std::mutex> lk(contextsMtx_);
  // Destroy in reverse order.
  for (auto it = contexts_.rbegin(); it != contexts_.rend(); ++it) {
    if (*it) {
      (*it)->destroy();
      it->reset();
    }
  }
  contexts_.clear();
}

bool VulkanDeviceContext::hasLiveContexts() {
  std::lock_guard<std::mutex> lk(contextsMtx_);
  return std::any_of(contexts_.begin(), contexts_.end(),
                     [](const auto& context) {
                       return context != nullptr;
                     });
}

// ── Constructor / Destructor ───────────────────────────────────────────────

VulkanDeviceContext::VulkanDeviceContext(int deviceId) : deviceId_(deviceId) {}

VulkanDeviceContext::~VulkanDeviceContext() {
  destroy();
}

// ── selectExtensions ───────────────────────────────────────────────────────

std::vector<const char*> VulkanDeviceContext::selectExtensions(
    const std::vector<VkExtensionProperties>& available)
{
  auto has = [&](const char* name) -> bool {
    for (auto& ext : available)
      if (strcmp(ext.extensionName, name) == 0) return true;
    return false;
  };

  std::vector<const char*> exts;
  const uint32_t apiVer = caps_.apiVersion;
  const bool features2Available =
      VulkanDeviceManager::getInstance().physicalDeviceFeatures2Fn() !=
      nullptr;

  // StorageBuffer storage class is core in Vulkan 1.1. Vulkan 1.0 devices
  // require the promoted KHR extension used by the SPIR-V target environment.
  if (apiVer < VK_API_VERSION_1_1 &&
      has(VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME)) {
    exts.push_back(VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
  }

  // 16-bit storage is core in Vulkan 1.1 and extension-backed in Vulkan 1.0.
  if (apiVer < VK_API_VERSION_1_1 && features2Available &&
      has(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
    exts.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
  }

#ifdef VK_KHR_8BIT_STORAGE_EXTENSION_NAME
  // 8-bit storage is core in Vulkan 1.2 and extension-backed before that.
  if (apiVer < VK_API_VERSION_1_2 && features2Available &&
      has(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
    exts.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
  }
#endif

  // Timeline semaphores: core in Vulkan 1.2+; extension in 1.0/1.1.
  if (apiVer < VK_API_VERSION_1_2 && features2Available &&
      has(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
    exts.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    caps_.timelineSem = true;
  }

  // Memory budget properties require a working Features2/Properties2 path.
  if (features2Available && has(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
    exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    caps_.memoryBudget = true;
  }

  // External memory host (zero-copy staging).
  if (has(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
    exts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    caps_.extMemHost = true;
  }

#if !defined(_WIN32)
  // Cross-device copies use one Vulkan-native primitive: opaque-FD external
  // memory plus an opaque-FD semaphore.  The pair is enabled atomically so no
  // caller can mistake a partial capability for peer-copy support. Vulkan 1.0
  // requires instance-level capability extensions that are not universally
  // available (notably on Android), so that path remains explicitly unsupported.
  if (apiVer >= VK_API_VERSION_1_1 &&
      has(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) &&
      has(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME)) {
    exts.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    caps_.externalMemoryFd = true;
    caps_.externalSemaphoreFd = true;
  }
#endif

  // fp16/int8 — extension in 1.0/1.1, core in 1.2.
  if (apiVer < VK_API_VERSION_1_2 && features2Available &&
      has(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
    exts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
  }

  return exts;
}

// ── Queue family selection ─────────────────────────────────────────────────

uint32_t VulkanDeviceContext::findComputeQueueFamily() const {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, nullptr);
  std::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, props.data());

  for (uint32_t i = 0; i < count; ++i) {
    if (props[i].queueCount > 0 &&
        (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
      return i;
    }
  }
  return UINT32_MAX;
}

uint32_t VulkanDeviceContext::findTransferQueueFamily(uint32_t computeFamily) const {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, nullptr);
  std::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, props.data());

  // Prefer a family that has TRANSFER but NOT COMPUTE (dedicated transfer engine).
  for (uint32_t i = 0; i < count; ++i) {
    if (i == computeFamily || props[i].queueCount == 0) continue;
    if ((props[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
        !(props[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
      return i;
  }
  // Use any other non-empty transfer family when no dedicated engine exists.
  for (uint32_t i = 0; i < count; ++i) {
    if (i != computeFamily && props[i].queueCount > 0 &&
        (props[i].queueFlags & VK_QUEUE_TRANSFER_BIT))
      return i;
  }
  return UINT32_MAX;  // will use compute queue for transfers
}

// ── initialize ─────────────────────────────────────────────────────────────

bool VulkanDeviceContext::initialize() {
  VulkanDeviceManager& mgr = VulkanDeviceManager::getInstance();
  physicalDevice_ = mgr.getPhysicalDevice(deviceId_);
  if (physicalDevice_ == VK_NULL_HANDLE) return false;

  const VulkanDeviceInfo* deviceInfo = mgr.getDeviceInfo(deviceId_);
  if (deviceInfo == nullptr || deviceInfo->apiVersionRaw == 0) return false;

  // Core feature availability is bounded by both the loader/instance API and
  // the selected physical device API (important on Android and older ICDs).
  uint32_t apiVer = std::min(mgr.negotiatedApiVersion(), deviceInfo->apiVersionRaw);
  caps_.apiVersion = apiVer;

  // ── Enumerate device extensions ─────────────────────────────────────
  std::vector<VkExtensionProperties> availExts;
  VkResult extensionResult = VK_SUCCESS;
  do {
    uint32_t extCount = 0;
    extensionResult = vkEnumerateDeviceExtensionProperties(
        physicalDevice_, nullptr, &extCount, nullptr);
    if (extensionResult != VK_SUCCESS) {
      sd_printf("VulkanDeviceContext: device-extension count failed "
                "device=%d res=%d\n",
                deviceId_, static_cast<int>(extensionResult));
      return false;
    }
    availExts.resize(extCount);
    extensionResult = vkEnumerateDeviceExtensionProperties(
        physicalDevice_, nullptr, &extCount,
        availExts.empty() ? nullptr : availExts.data());
    availExts.resize(extCount);
  } while (extensionResult == VK_INCOMPLETE);
  if (extensionResult != VK_SUCCESS) {
    sd_printf("VulkanDeviceContext: device-extension enumeration failed "
              "device=%d res=%d\n",
              deviceId_, static_cast<int>(extensionResult));
    return false;
  }

  if (apiVer < VK_API_VERSION_1_1) {
    const bool hasStorageBufferStorageClass = std::any_of(
        availExts.begin(), availExts.end(), [](const VkExtensionProperties& ext) {
          return strcmp(
                     ext.extensionName,
                     VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME) == 0;
        });
    if (!hasStorageBufferStorageClass) {
      sd_printf("VulkanDeviceContext: Vulkan 1.0 device %d lacks required "
                "VK_KHR_storage_buffer_storage_class\n", deviceId_);
      return false;
    }
  }

  std::vector<const char*> enabledExts = selectExtensions(availExts);
  auto extensionEnabled = [&](const char* name) {
    return std::any_of(enabledExts.begin(), enabledExts.end(),
                       [&](const char* enabled) {
                         return strcmp(enabled, name) == 0;
                       });
  };

  // Promoted feature structures may only be placed in the pNext chain when
  // provided by the negotiated core version or its enabled device extension.
  const bool hasStorage16FeatureStruct =
      apiVer >= VK_API_VERSION_1_1 ||
      extensionEnabled(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
  const bool hasFloat16Int8FeatureStruct =
      apiVer >= VK_API_VERSION_1_2 ||
      extensionEnabled(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
#ifdef VK_KHR_8BIT_STORAGE_EXTENSION_NAME
  const bool hasStorage8FeatureStruct =
      apiVer >= VK_API_VERSION_1_2 ||
      extensionEnabled(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
#else
  const bool hasStorage8FeatureStruct = false;
#endif
  const auto getPhysicalDeviceFeatures2 =
      mgr.physicalDeviceFeatures2Fn();

  // Android's API-24 Vulkan stub does not export Vulkan 1.1 entry points as
  // link-time symbols. Resolve instance commands through the loader so one
  // backend binary remains compatible with API 24 while using 1.1 features
  // when the device and loader actually provide them.
  const auto getPhysicalDeviceProperties2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
          vkGetInstanceProcAddr(mgr.getInstance_(),
                                "vkGetPhysicalDeviceProperties2"));
  const auto getPhysicalDeviceExternalBufferProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceExternalBufferProperties>(
          vkGetInstanceProcAddr(
              mgr.getInstance_(),
              "vkGetPhysicalDeviceExternalBufferProperties"));
  const auto getPhysicalDeviceExternalSemaphoreProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceExternalSemaphoreProperties>(
          vkGetInstanceProcAddr(
              mgr.getInstance_(),
              "vkGetPhysicalDeviceExternalSemaphoreProperties"));

  // ── Find queue families ─────────────────────────────────────────────
  caps_.computeQueueFamily  = findComputeQueueFamily();
  caps_.transferQueueFamily = findTransferQueueFamily(caps_.computeQueueFamily);

  if (caps_.computeQueueFamily == UINT32_MAX) {
    sd_printf("VulkanDeviceContext: device %d has no compute queue family\n", deviceId_);
    return false;
  }

  // ── Query device features ────────────────────────────────────────────
  // Chain feature structs.  Use value-based MLIR guard.
  VkPhysicalDeviceFeatures2 features2 = {};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

  VkPhysicalDevice16BitStorageFeatures storage16Features = {};
  storage16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

#ifdef VK_KHR_8BIT_STORAGE_EXTENSION_NAME
  VkPhysicalDevice8BitStorageFeatures storage8Features = {};
  storage8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
#endif

  VkPhysicalDeviceShaderFloat16Int8Features fp16int8Features = {};
  fp16int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

  VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = {};
  timelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;

  // Query the complete feature chain that may be enabled on this logical
  // device. A physical-device feature is not a backend capability until it is
  // also present in the VkDeviceCreateInfo chain below.
  const bool queryTimeline =
      apiVer >= VK_API_VERSION_1_2 || caps_.timelineSem;
  if (getPhysicalDeviceFeatures2 != nullptr) {
    void** queryTail = &features2.pNext;
    if (hasStorage16FeatureStruct) {
      *queryTail = &storage16Features;
      queryTail = &storage16Features.pNext;
    }
#ifdef VK_KHR_8BIT_STORAGE_EXTENSION_NAME
    if (hasStorage8FeatureStruct) {
      *queryTail = &storage8Features;
      queryTail = &storage8Features.pNext;
    }
#endif
    if (hasFloat16Int8FeatureStruct) {
      *queryTail = &fp16int8Features;
      queryTail = &fp16int8Features.pNext;
    }
    if (queryTimeline) {
      *queryTail = &timelineFeatures;
    }

    getPhysicalDeviceFeatures2(physicalDevice_, &features2);
    caps_.fp16 = hasFloat16Int8FeatureStruct &&
                 fp16int8Features.shaderFloat16 == VK_TRUE;
    caps_.fp64 = features2.features.shaderFloat64 == VK_TRUE;
    caps_.int64 = features2.features.shaderInt64 == VK_TRUE;
    caps_.int8 = hasFloat16Int8FeatureStruct &&
                 fp16int8Features.shaderInt8 == VK_TRUE;
    caps_.storage16 = hasStorage16FeatureStruct &&
                      storage16Features.storageBuffer16BitAccess == VK_TRUE;
#ifdef VK_KHR_8BIT_STORAGE_EXTENSION_NAME
    caps_.storage8 = hasStorage8FeatureStruct &&
                     storage8Features.storageBuffer8BitAccess == VK_TRUE;
#endif
    if (queryTimeline) {
      caps_.timelineSem = timelineFeatures.timelineSemaphore == VK_TRUE;
    }
  } else {
    // Vulkan 1.0 without KHR_get_physical_device_properties2 still exposes
    // all core 1.0 features through the legacy query.
    VkPhysicalDeviceFeatures coreFeatures = {};
    vkGetPhysicalDeviceFeatures(physicalDevice_, &coreFeatures);
    caps_.fp64 = coreFeatures.shaderFloat64 == VK_TRUE;
    caps_.int64 = coreFeatures.shaderInt64 == VK_TRUE;
  }

  // Verify that the enabled opaque-FD extension pair supports the exact
  // buffer/semaphore handle types used by peer copies. Extension presence alone
  // is not a capability guarantee.
  if (caps_.externalMemoryFd && caps_.externalSemaphoreFd &&
      getPhysicalDeviceExternalBufferProperties != nullptr &&
      getPhysicalDeviceExternalSemaphoreProperties != nullptr) {
    VkPhysicalDeviceExternalBufferInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    bufferInfo.flags = 0;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkExternalBufferProperties bufferProperties = {};
    bufferProperties.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    getPhysicalDeviceExternalBufferProperties(
        physicalDevice_, &bufferInfo, &bufferProperties);

    VkPhysicalDeviceExternalSemaphoreInfo semaphoreInfo = {};
    semaphoreInfo.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    semaphoreInfo.handleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkExternalSemaphoreProperties semaphoreProperties = {};
    semaphoreProperties.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    getPhysicalDeviceExternalSemaphoreProperties(
        physicalDevice_, &semaphoreInfo, &semaphoreProperties);

    const VkExternalMemoryFeatureFlags memoryFeatures =
        bufferProperties.externalMemoryProperties.externalMemoryFeatures;
    const VkExternalSemaphoreFeatureFlags semaphoreFeatures =
        semaphoreProperties.externalSemaphoreFeatures;
    const bool memoryUsable =
        (memoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0 &&
        (memoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0 &&
        (bufferProperties.externalMemoryProperties.compatibleHandleTypes &
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) != 0;
    const bool semaphoreUsable =
        (semaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0 &&
        (semaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0 &&
        (semaphoreProperties.compatibleHandleTypes &
         VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) != 0;
    if (!memoryUsable || !semaphoreUsable) {
      caps_.externalMemoryFd = false;
      caps_.externalSemaphoreFd = false;
    } else {
      caps_.externalMemoryDedicatedOnly =
          (memoryFeatures &
           VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
    }
  } else if (caps_.externalMemoryFd && caps_.externalSemaphoreFd) {
    // The driver advertised peer-copy extensions but its loader did not expose
    // the corresponding capability queries. Treat the optional feature as
    // unavailable rather than making context initialization unsafe.
    caps_.externalMemoryFd = false;
    caps_.externalSemaphoreFd = false;
  }

  // Query subgroup size (Vulkan 1.1+).
  if (apiVer >= VK_API_VERSION_1_1 &&
      getPhysicalDeviceProperties2 != nullptr) {
    VkPhysicalDeviceSubgroupProperties sgProps = {};
    sgProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &sgProps;
    getPhysicalDeviceProperties2(physicalDevice_, &props2);
    caps_.subgroupSize = sgProps.subgroupSize;
  }

  sd_printf("VulkanDeviceContext: device=%d fp16=%s storage16=%s storage8=%s fp64=%s int8=%s timeline=%s externalPeer=%s subgroup=%u\n",
            deviceId_,
            caps_.fp16 ? "yes" : "no",
            caps_.storage16 ? "yes" : "no",
            caps_.storage8 ? "yes" : "no",
            caps_.fp64 ? "yes" : "no",
            caps_.int8 ? "yes" : "no",
            caps_.timelineSem ? "yes" : "no",
            (caps_.externalMemoryFd && caps_.externalSemaphoreFd) ? "yes" : "no",
            caps_.subgroupSize);

  // ── Build VkDeviceCreateInfo ─────────────────────────────────────────
  float priority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queueInfos;

  VkDeviceQueueCreateInfo computeQI = {};
  computeQI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  computeQI.queueFamilyIndex = caps_.computeQueueFamily;
  computeQI.queueCount       = 1;
  computeQI.pQueuePriorities = &priority;
  queueInfos.push_back(computeQI);

  if (caps_.transferQueueFamily != UINT32_MAX &&
      caps_.transferQueueFamily != caps_.computeQueueFamily) {
    VkDeviceQueueCreateInfo transferQI = {};
    transferQI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    transferQI.queueFamilyIndex = caps_.transferQueueFamily;
    transferQI.queueCount       = 1;
    transferQI.pQueuePriorities = &priority;
    queueInfos.push_back(transferQI);
  }

  // Feature enablement structs for the logical device.
  VkPhysicalDeviceFeatures2 enableFeatures2 = {};
  enableFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  enableFeatures2.features.shaderFloat64 = caps_.fp64 ? VK_TRUE : VK_FALSE;
  enableFeatures2.features.shaderInt64 = caps_.int64 ? VK_TRUE : VK_FALSE;

  VkPhysicalDevice16BitStorageFeatures enableStorage16 = {};
  enableStorage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  enableStorage16.storageBuffer16BitAccess = caps_.storage16 ? VK_TRUE : VK_FALSE;

#ifdef VK_KHR_8BIT_STORAGE_EXTENSION_NAME
  VkPhysicalDevice8BitStorageFeatures enableStorage8 = {};
  enableStorage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
  enableStorage8.storageBuffer8BitAccess = caps_.storage8 ? VK_TRUE : VK_FALSE;
#endif

  VkPhysicalDeviceShaderFloat16Int8Features enableFp16 = {};
  enableFp16.sType          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  enableFp16.shaderFloat16  = caps_.fp16 ? VK_TRUE : VK_FALSE;
  enableFp16.shaderInt8     = caps_.int8 ? VK_TRUE : VK_FALSE;

  VkPhysicalDeviceTimelineSemaphoreFeatures enableTimeline = {};
  enableTimeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  enableTimeline.timelineSemaphore = caps_.timelineSem ? VK_TRUE : VK_FALSE;

  VkDeviceCreateInfo devCI = {};
  devCI.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  devCI.queueCreateInfoCount    = static_cast<uint32_t>(queueInfos.size());
  devCI.pQueueCreateInfos       = queueInfos.data();
  devCI.enabledExtensionCount   = static_cast<uint32_t>(enabledExts.size());
  devCI.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();
  devCI.enabledLayerCount       = 0;

  // Enable exactly the features reported through VulkanDeviceCaps, using
  // only feature structures legal for the negotiated core/extension path.
  VkPhysicalDeviceFeatures enableCoreFeatures = {};
  enableCoreFeatures.shaderFloat64 = caps_.fp64 ? VK_TRUE : VK_FALSE;
  enableCoreFeatures.shaderInt64 = caps_.int64 ? VK_TRUE : VK_FALSE;
  if (getPhysicalDeviceFeatures2 != nullptr) {
    devCI.pNext = &enableFeatures2;
    void** enableTail = &enableFeatures2.pNext;
    if (hasStorage16FeatureStruct) {
      *enableTail = &enableStorage16;
      enableTail = &enableStorage16.pNext;
    }
#ifdef VK_KHR_8BIT_STORAGE_EXTENSION_NAME
    if (hasStorage8FeatureStruct) {
      *enableTail = &enableStorage8;
      enableTail = &enableStorage8.pNext;
    }
#endif
    if (hasFloat16Int8FeatureStruct) {
      *enableTail = &enableFp16;
      enableTail = &enableFp16.pNext;
    }
    if (caps_.timelineSem) {
      *enableTail = &enableTimeline;
    }
  } else {
    devCI.pEnabledFeatures = &enableCoreFeatures;
  }

  VkResult res = vkCreateDevice(physicalDevice_, &devCI, nullptr, &device_);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanDeviceContext: vkCreateDevice failed device=%d res=%d\n",
              deviceId_, static_cast<int>(res));
    return false;
  }

  // Resolve timeline entry points through the exact capability enabled above.
  // Vulkan 1.1 extension devices (common on older AMD drivers and Android)
  // expose only the KHR-suffixed symbols; Vulkan 1.2+ exposes the core symbols.
  if (caps_.timelineSem) {
    const bool coreTimeline = apiVer >= VK_API_VERSION_1_2;
    signalSemaphoreFn_ = reinterpret_cast<PFN_vkSignalSemaphore>(
        vkGetDeviceProcAddr(device_,
                            coreTimeline ? "vkSignalSemaphore"
                                         : "vkSignalSemaphoreKHR"));
    waitSemaphoresFn_ = reinterpret_cast<PFN_vkWaitSemaphores>(
        vkGetDeviceProcAddr(device_,
                            coreTimeline ? "vkWaitSemaphores"
                                         : "vkWaitSemaphoresKHR"));
    getSemaphoreCounterValueFn_ =
        reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
            vkGetDeviceProcAddr(
                device_, coreTimeline ? "vkGetSemaphoreCounterValue"
                                      : "vkGetSemaphoreCounterValueKHR"));
    if (signalSemaphoreFn_ == nullptr || waitSemaphoresFn_ == nullptr ||
        getSemaphoreCounterValueFn_ == nullptr) {
      sd_printf("VulkanDeviceContext: device %d advertises timeline semaphores "
                "but the negotiated %s entry points are unavailable\n",
                deviceId_, coreTimeline ? "Vulkan 1.2 core"
                                        : "VK_KHR_timeline_semaphore");
      return false;
    }
  }

  if (caps_.externalMemoryFd && caps_.externalSemaphoreFd) {
    getMemoryFdFn_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
    getMemoryFdPropertiesFn_ =
        reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryFdPropertiesKHR"));
    getSemaphoreFdFn_ = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device_, "vkGetSemaphoreFdKHR"));
    importSemaphoreFdFn_ =
        reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
            vkGetDeviceProcAddr(device_, "vkImportSemaphoreFdKHR"));
    if (getMemoryFdFn_ == nullptr || getMemoryFdPropertiesFn_ == nullptr ||
        getSemaphoreFdFn_ == nullptr || importSemaphoreFdFn_ == nullptr) {
      sd_printf("VulkanDeviceContext: device %d enabled external FD sharing "
                "but one or more device entry points are unavailable; "
                "cross-device copy is disabled\n",
                deviceId_);
      caps_.externalMemoryFd = false;
      caps_.externalSemaphoreFd = false;
      getMemoryFdFn_ = nullptr;
      getMemoryFdPropertiesFn_ = nullptr;
      getSemaphoreFdFn_ = nullptr;
      importSemaphoreFdFn_ = nullptr;
    }
  }

  // ── Get queues ────────────────────────────────────────────────────────
  vkGetDeviceQueue(device_, caps_.computeQueueFamily, 0, &computeQueue_);
  if (caps_.transferQueueFamily != UINT32_MAX &&
      caps_.transferQueueFamily != caps_.computeQueueFamily) {
    vkGetDeviceQueue(device_, caps_.transferQueueFamily, 0, &transferQueue_);
  }

  // ── Create descriptor pool arena ──────────────────────────────────────
  {
    std::vector<VkDescriptorPoolSize> poolSizes = {
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 },
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256  },
    };
    VkDescriptorPoolCreateInfo dpCI = {};
    dpCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpCI.maxSets       = 512;
    dpCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    dpCI.pPoolSizes    = poolSizes.data();
    dpCI.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(device_, &dpCI, nullptr, &descriptorPool_) != VK_SUCCESS) {
      sd_printf("VulkanDeviceContext: vkCreateDescriptorPool failed device=%d\n", deviceId_);
      // Non-fatal for now; pool will be null.
    }
  }

  // ── Create pipeline cache ─────────────────────────────────────────────
  {
    VkPipelineCacheCreateInfo pcCI = {};
    pcCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    // Tier-2 (ADR 0115): seed from the persisted per-device blob so the
    // driver skips SPIR-V→ISA recompilation for previously seen shaders.
    std::vector<char> blob;
    if (sd::config::VulkanConfig::getInstance().pipelineCacheEnabled()) {
      blob = readPipelineCacheBlob(physicalDevice_);
      if (!blob.empty()) {
        pcCI.initialDataSize = blob.size();
        pcCI.pInitialData = blob.data();
      }
    }
    VkResult pcRes = vkCreatePipelineCache(device_, &pcCI, nullptr, &pipelineCache_);
    if (pcRes != VK_SUCCESS && !blob.empty()) {
      // Retry once without initial data — never let a stale blob keep the
      // device from getting a pipeline cache at all.
      pipelineCache_ = VK_NULL_HANDLE;
      pcCI.initialDataSize = 0;
      pcCI.pInitialData = nullptr;
      blob.clear();
      pcRes = vkCreatePipelineCache(device_, &pcCI, nullptr, &pipelineCache_);
    }
    if (pcRes != VK_SUCCESS) {
      sd_printf("VulkanDeviceContext: vkCreatePipelineCache failed device=%d\n", deviceId_);
    } else if (!blob.empty()) {
      lastSavedPipelineCacheBytes_ = blob.size();
      sd::config::VulkanConfig::getInstance().incrementPipelineBlobLoads();
      DSP_DIAG(BACKEND, "VulkanDeviceContext: pipeline-cache blob loaded (%zu bytes) device=%d",
               blob.size(), deviceId_);
    }
  }

  // Native eager kernels and offline bundle dispatch share one shader
  // pipeline cache for the lifetime of this logical device. MLIR compilation
  // is optional; precompiled SPIR-V loading remains available on mobile.
  shaderPipelineCache_ =
      std::make_unique<VulkanPipelineCache>(device_, physicalDevice_, caps_,
                                            pipelineCache_);

  // ── Create timeline semaphore (ordering spine) ────────────────────────
  if (caps_.timelineSem) {
    VkSemaphoreTypeCreateInfo semTypeCI = {};
    semTypeCI.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semTypeCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semTypeCI.initialValue  = 0;

    VkSemaphoreCreateInfo semCI = {};
    semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semCI.pNext = &semTypeCI;

    VkResult timelineResult =
        vkCreateSemaphore(device_, &semCI, nullptr, &timelineSemaphore_);
    if (timelineResult != VK_SUCCESS) {
      sd_printf("VulkanDeviceContext: vkCreateSemaphore (timeline) failed "
                "device=%d res=%d\n",
                deviceId_, static_cast<int>(timelineResult));
      return false;
    }
    sd_printf("VulkanDeviceContext: timeline semaphore created device=%d\n",
              deviceId_);
  }

  sd_printf("VulkanDeviceContext: initialized device=%d extCount=%zu transferQueue=%s timeline=%s\n",
            deviceId_,
            enabledExts.size(),
            transferQueue_ != VK_NULL_HANDLE ? "dedicated" : "compute-shared",
            timelineSemaphore_ != VK_NULL_HANDLE ? "yes" : "no");
  return true;
}

// ── destroy ────────────────────────────────────────────────────────────────

void VulkanDeviceContext::destroy() {
  if (device_ == VK_NULL_HANDLE) return;

  {
    std::lock_guard<std::mutex> queueLock(computeQueueMtx_);
    VkResult idleResult = vkDeviceWaitIdle(device_);
    if (idleResult != VK_SUCCESS) {
      sd_printf("VulkanDeviceContext::destroy: vkDeviceWaitIdle failed "
                "device=%d res=%d\n",
                deviceId_, static_cast<int>(idleResult));
    }
  }

  // Free all per-thread command pools.
  {
    std::lock_guard<std::mutex> lk(cmdPoolMtx_);
    for (auto& [tid, pool] : cmdPools_) {
      if (pool != VK_NULL_HANDLE) vkDestroyCommandPool(device_, pool, nullptr);
    }
    cmdPools_.clear();
  }

  {
    std::lock_guard<std::mutex> fenceLock(unresolvedFenceMtx_);
    for (VkFence fence : unresolvedFences_) {
      if (fence != VK_NULL_HANDLE) vkDestroyFence(device_, fence, nullptr);
    }
    unresolvedFences_.clear();
  }

  shaderPipelineCache_.reset();

  if (timelineSemaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(device_, timelineSemaphore_, nullptr);
    timelineSemaphore_ = VK_NULL_HANDLE;
  }
  if (pipelineCache_ != VK_NULL_HANDLE) {
    // Tier-2 (ADR 0115): persist the accumulated driver blob before the
    // handle goes away. The device is idle (vkDeviceWaitIdle above).
    savePipelineCacheBlob();
    vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
    pipelineCache_ = VK_NULL_HANDLE;
  }
  {
    std::lock_guard<std::mutex> descriptorLock(descriptorPoolMtx_);
    if (descriptorPool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
      descriptorPool_ = VK_NULL_HANDLE;
    }
  }

  vkDestroyDevice(device_, nullptr);
  device_ = VK_NULL_HANDLE;
  signalSemaphoreFn_ = nullptr;
  waitSemaphoresFn_ = nullptr;
  getSemaphoreCounterValueFn_ = nullptr;
  getMemoryFdFn_ = nullptr;
  getMemoryFdPropertiesFn_ = nullptr;
  getSemaphoreFdFn_ = nullptr;
  importSemaphoreFdFn_ = nullptr;
}

// ── Tier-2 pipeline-cache blob persistence (ADR 0115) ──────────────────────

bool VulkanDeviceContext::savePipelineCacheBlob() {
  auto& cfg = sd::config::VulkanConfig::getInstance();
  if (!cfg.pipelineCacheEnabled()) return false;
  if (device_ == VK_NULL_HANDLE || pipelineCache_ == VK_NULL_HANDLE) return false;
  if (isLost()) return false;

  std::lock_guard<std::mutex> lk(pipelineCacheBlobMtx_);

  size_t dataSize = 0;
  if (vkGetPipelineCacheData(device_, pipelineCache_, &dataSize, nullptr) != VK_SUCCESS ||
      dataSize == 0) {
    return false;
  }
  // Flush heuristic (ADR 0115): an unchanged size since the last write means
  // no new pipelines were compiled — skip the disk write on hot paths.
  if (dataSize == lastSavedPipelineCacheBytes_) return true;

  const int64_t maxBytes = cfg.pipelineCacheMaxBytes();
  if (maxBytes > 0 && static_cast<int64_t>(dataSize) > maxBytes) {
    DSP_DIAG(BACKEND,
             "VulkanDeviceContext: pipeline-cache blob (%zu bytes) exceeds budget (%lld) — not saved",
             dataSize, static_cast<long long>(maxBytes));
    return false;
  }

  std::vector<char> data(dataSize);
  if (vkGetPipelineCacheData(device_, pipelineCache_, &dataSize, data.data()) != VK_SUCCESS) {
    // The cache can grow between the size query and the data query
    // (VK_INCOMPLETE) — skip this flush; the next one picks it up.
    return false;
  }
  data.resize(dataSize);

  const std::string path = pipelineCacheBlobPath(physicalDevice_);
  const size_t slash = path.find_last_of('/');
  const std::string dir = (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
  if (!VulkanSpirvDiskCache::ensureDir(dir)) return false;
  if (!VulkanSpirvDiskCache::atomicWrite(path, data.data(), data.size())) return false;

  lastSavedPipelineCacheBytes_ = data.size();
  cfg.incrementPipelineBlobSaves();
  DSP_DIAG(BACKEND, "VulkanDeviceContext: pipeline-cache blob saved (%zu bytes) device=%d",
           data.size(), deviceId_);
  return true;
}

// ── Canonical queue synchronization ────────────────────────────────────────

VkResult VulkanDeviceContext::submitCompute(
    uint32_t submitCount, const VkSubmitInfo* submitInfos, VkFence fence) {
  if (computeQueue_ == VK_NULL_HANDLE || submitCount == 0 ||
      submitInfos == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  std::lock_guard<std::mutex> queueLock(computeQueueMtx_);
  return vkQueueSubmit(computeQueue_, submitCount, submitInfos, fence);
}

VkResult VulkanDeviceContext::waitComputeIdle() {
  if (computeQueue_ == VK_NULL_HANDLE) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  std::lock_guard<std::mutex> queueLock(computeQueueMtx_);
  return vkQueueWaitIdle(computeQueue_);
}

VkResult VulkanDeviceContext::allocateDescriptorSets(
    const VkDescriptorSetAllocateInfo* allocateInfo,
    VkDescriptorSet* descriptorSets) {
  if (device_ == VK_NULL_HANDLE || descriptorPool_ == VK_NULL_HANDLE ||
      allocateInfo == nullptr || descriptorSets == nullptr ||
      allocateInfo->descriptorPool != descriptorPool_) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  std::lock_guard<std::mutex> descriptorLock(descriptorPoolMtx_);
  return vkAllocateDescriptorSets(device_, allocateInfo, descriptorSets);
}

VkResult VulkanDeviceContext::freeDescriptorSets(
    uint32_t descriptorSetCount, const VkDescriptorSet* descriptorSets) {
  if (device_ == VK_NULL_HANDLE || descriptorPool_ == VK_NULL_HANDLE ||
      descriptorSetCount == 0 || descriptorSets == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  std::lock_guard<std::mutex> descriptorLock(descriptorPoolMtx_);
  return vkFreeDescriptorSets(device_, descriptorPool_, descriptorSetCount,
                              descriptorSets);
}

// ── Per-thread command pools ───────────────────────────────────────────────

VkCommandPool VulkanDeviceContext::getThreadCommandPool() {
  std::thread::id tid = std::this_thread::get_id();
  {
    std::lock_guard<std::mutex> lk(cmdPoolMtx_);
    auto it = cmdPools_.find(tid);
    if (it != cmdPools_.end()) return it->second;
  }
  // Create a new pool for this thread.
  VkCommandPoolCreateInfo cpCI = {};
  cpCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpCI.queueFamilyIndex = caps_.computeQueueFamily;
  cpCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  VkCommandPool pool = VK_NULL_HANDLE;
  if (vkCreateCommandPool(device_, &cpCI, nullptr, &pool) != VK_SUCCESS) {
    sd_printf("VulkanDeviceContext::getThreadCommandPool: vkCreateCommandPool failed device=%d\n",
              deviceId_);
    return VK_NULL_HANDLE;
  }
  {
    std::lock_guard<std::mutex> lk(cmdPoolMtx_);
    cmdPools_[tid] = pool;
  }
  return pool;
}

void VulkanDeviceContext::releaseThreadCommandPool() {
  std::thread::id tid = std::this_thread::get_id();
  VkCommandPool pool = VK_NULL_HANDLE;
  {
    std::lock_guard<std::mutex> lk(cmdPoolMtx_);
    auto it = cmdPools_.find(tid);
    if (it == cmdPools_.end()) return;
    pool = it->second;
    cmdPools_.erase(it);
  }
  if (pool != VK_NULL_HANDLE) vkDestroyCommandPool(device_, pool, nullptr);
}

// ── Timeline semaphore ops ─────────────────────────────────────────────────

void VulkanDeviceContext::signalTimeline(uint64_t value) {
  if (timelineSemaphore_ == VK_NULL_HANDLE || signalSemaphoreFn_ == nullptr)
    return;

  std::lock_guard<std::mutex> signalLock(timelineSignalMtx_);
  uint64_t currentValue = 0;
  if (getSemaphoreCounterValueFn_ != nullptr &&
      getSemaphoreCounterValueFn_(device_, timelineSemaphore_,
                                  &currentValue) == VK_SUCCESS &&
      currentValue >= value) {
    return;
  }

  VkSemaphoreSignalInfo signalInfo = {};
  signalInfo.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signalInfo.semaphore = timelineSemaphore_;
  signalInfo.value     = value;

  VkResult res = signalSemaphoreFn_(device_, &signalInfo);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanDeviceContext::signalTimeline: signal failed device=%d val=%llu res=%d\n",
              deviceId_, (unsigned long long)value, static_cast<int>(res));
    if (res == VK_ERROR_DEVICE_LOST) markLost();
  }
}

bool VulkanDeviceContext::waitTimeline(uint64_t value, uint64_t timeoutNs) {
  if (timelineSemaphore_ == VK_NULL_HANDLE || waitSemaphoresFn_ == nullptr)
    return false;

  VkSemaphoreWaitInfo waitInfo = {};
  waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  waitInfo.semaphoreCount = 1;
  waitInfo.pSemaphores    = &timelineSemaphore_;
  waitInfo.pValues        = &value;

  VkResult res = waitSemaphoresFn_(device_, &waitInfo, timeoutNs);
  if (res == VK_SUCCESS) return true;
  if (res == VK_TIMEOUT) return false;
  sd_printf("VulkanDeviceContext::waitTimeline: wait failed device=%d val=%llu res=%d\n",
            deviceId_, (unsigned long long)value, static_cast<int>(res));
  if (res == VK_ERROR_DEVICE_LOST) markLost();
  return false;
}

uint64_t VulkanDeviceContext::queryTimeline() const {
  if (timelineSemaphore_ == VK_NULL_HANDLE ||
      getSemaphoreCounterValueFn_ == nullptr)
    return 0;
  uint64_t value = 0;
  VkResult res =
      getSemaphoreCounterValueFn_(device_, timelineSemaphore_, &value);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanDeviceContext::queryTimeline: query failed device=%d res=%d\n",
              deviceId_, static_cast<int>(res));
    return 0;
  }
  return value;
}

// ── submitAndSweep ─────────────────────────────────────────────────────────

void VulkanDeviceContext::submitAndSweep(VkSubmitInfo submitInfo, uint64_t timelineValue) {
  if (lost_.load(std::memory_order_acquire)) {
    sd_printf("VulkanDeviceContext::submitAndSweep: device %d is lost, rejecting submit\n", deviceId_);
    return;
  }

  // Create a binary fence to signal submission completion.
  VkFenceCreateInfo fCI = {};
  fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  VkResult res = vkCreateFence(device_, &fCI, nullptr, &fence);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanDeviceContext::submitAndSweep: vkCreateFence failed "
              "device=%d res=%d\n",
              deviceId_, static_cast<int>(res));
    if (res == VK_ERROR_DEVICE_LOST) markLost();
    return;
  }

  res = submitCompute(1, &submitInfo, fence);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanDeviceContext::submitAndSweep: vkQueueSubmit failed device=%d res=%d\n",
              deviceId_, static_cast<int>(res));
    if (res == VK_ERROR_DEVICE_LOST) markLost();
    if (fence != VK_NULL_HANDLE) vkDestroyFence(device_, fence, nullptr);
    return;
  }

  // This method is synchronous by contract: never advance the retirement
  // timeline until the submitted work is known to have completed.
  VkResult waitResult =
      vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
  if (waitResult != VK_SUCCESS) {
    sd_printf("VulkanDeviceContext::submitAndSweep: vkWaitForFences failed "
              "device=%d res=%d\n",
              deviceId_, static_cast<int>(waitResult));
    markLost();
    {
      std::lock_guard<std::mutex> fenceLock(unresolvedFenceMtx_);
      unresolvedFences_.push_back(fence);
    }
    return;
  }
  vkDestroyFence(device_, fence, nullptr);

  // Signal the timeline semaphore (host-side: tells the pool this epoch passed).
  if (timelineSemaphore_ != VK_NULL_HANDLE) {
    signalTimeline(timelineValue);
  }

  // Sweep the memory pool for this device.
  uint64_t completedTimeline = timelineSemaphore_ != VK_NULL_HANDLE
                               ? queryTimeline()
                               : timelineValue;
  VulkanMemoryPool::getInstance().sweep(deviceId_, completedTimeline);

  DSP_DIAG(MEMORY,
    "VulkanDeviceContext::submitAndSweep device=%d timeline=%llu swept",
    deviceId_, (unsigned long long)completedTimeline);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
