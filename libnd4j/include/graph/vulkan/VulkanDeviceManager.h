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

#ifndef LIBND4J_VULKAN_DEVICE_MANAGER_H
#define LIBND4J_VULKAN_DEVICE_MANAGER_H

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * Per-device information cached at enumeration time.
 */
struct VulkanDeviceInfo {
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  std::string name;
  uint32_t apiVersionRaw = 0;     // raw VK_MAKE_VERSION value
  uint32_t vendorId = 0;          // PCI/vendor ID reported by the Vulkan ICD
  int vkMajor = 0;                // VK_VERSION_MAJOR(apiVersionRaw)
  int vkMinor = 0;                // VK_VERSION_MINOR(apiVersionRaw)
  uint64_t totalMemoryBytes = 0;  // largest DEVICE_LOCAL heap
  bool memoryBudgetExt = false;   // VK_EXT_memory_budget available
};

/**
 * Singleton managing shared VkInstance and enumerated physical devices
 * for the SD_VULKAN chip build.
 *
 * Design contract (mirrors CUDA NativeOps reference):
 *  - getAvailableDevices()     : returns device count, 0 on Vulkan init failure
 *  - getDevice()               : thread-local current device id (default 0)
 *  - setDevice(id)             : sets thread-local current device
 *  - getDeviceName(id)         : VkPhysicalDeviceProperties::deviceName
 *  - getDeviceMajor(id)        : VK_VERSION_MAJOR(apiVersion)
 *  - getDeviceMinor(id)        : VK_VERSION_MINOR(apiVersion)
 *  - getDeviceTotalMemory(id)  : largest DEVICE_LOCAL heap size
 *  - getDeviceFreeMemory(id)   : total minus tracked allocs; with
 *                                VK_EXT_memory_budget uses driver budget
 *  - getDeviceFreeMemoryDefault(): free memory on current device
 *
 * Logical-device ownership:
 *   VulkanDeviceManager owns only the shared VkInstance and physical-device
 *   enumeration. VulkanDeviceContext is the single canonical owner of the
 *   feature-enabled VkDevice for each device id; memory pools, queues, and
 *   replay all borrow that same logical device.
 */
class SD_LIB_EXPORT VulkanDeviceManager {
 public:
  /** Returns the singleton instance. Thread-safe (Meyers singleton). */
  static VulkanDeviceManager& getInstance();

  // Non-copyable, non-movable
  VulkanDeviceManager(const VulkanDeviceManager&) = delete;
  VulkanDeviceManager& operator=(const VulkanDeviceManager&) = delete;

  /**
   * Initialize (or no-op if already initialized).
   * Returns true when at least one Vulkan device is available.
   * Safe to call from multiple threads — guarded by mutex.
   */
  bool initialize();

  /** True when initialize() succeeded and deviceCount() > 0. */
  bool isInitialized() const;

  /**
   * Destroy the shared VkInstance after canonical device contexts and their
   * memory pools have been shut down. Safe to call repeatedly.
   */
  void shutdown();

  // ── Device enumeration ──────────────────────────────────────────────

  /** Number of enumerated physical devices (0 if Vulkan unavailable). */
  int deviceCount() const;

  /** Physical device handle for deviceId, or VK_NULL_HANDLE if OOB. */
  VkPhysicalDevice getPhysicalDevice(int deviceId) const;

  /** Cached info struct for deviceId. Only valid when isInitialized(). */
  const VulkanDeviceInfo* getDeviceInfo(int deviceId) const;

  // ── Thread-local current device ──────────────────────────────────────

  /** Thread-local current device id (default 0). */
  static int currentDeviceId();

  /** Set thread-local current device (returns false if OOB). */
  bool setCurrentDevice(int deviceId);

  // ── Memory budget tracking ─────────────────────────────────────────

  /**
   * Track an allocation against a device so that getDeviceFreeMemory()
   * can approximate free memory on hardware without VK_EXT_memory_budget.
   */
  void trackAllocation(int deviceId, size_t bytes);

  /**
   * Untrack an allocation against a device.
   * If bytes > tracked, clamps to 0 (defensive against double-accounting).
   */
  void untrackAllocation(int deviceId, size_t bytes);

  /**
   * Return approximate free memory for deviceId.
   * Uses VK_EXT_memory_budget when available, otherwise
   * total – tracked_allocs.
   */
  uint64_t getFreeMemory(int deviceId) const;

  /** The shared VkInstance (or VK_NULL_HANDLE before initialize()). */
  VkInstance getInstance_() const { return instance_; }

  /**
   * The Vulkan API version that was successfully negotiated during
   * initialize() — VK_API_VERSION_1_2, VK_API_VERSION_1_1, or
   * VK_API_VERSION_1_0.  Returns 0 before initialize() is called.
   */
  uint32_t negotiatedApiVersion() const { return negotiatedApiVersion_; }

  /** Features2 query entry point for the negotiated core/KHR instance path. */
  PFN_vkGetPhysicalDeviceFeatures2 physicalDeviceFeatures2Fn() const {
    return physicalDeviceFeatures2Fn_;
  }

 private:
  VulkanDeviceManager();
  ~VulkanDeviceManager();

  bool doInitialize();

  // ── State ───────────────────────────────────────────────────────────

  mutable std::mutex mutex_;
  bool initialized_ = false;
  bool initFailed_ = false;

  VkInstance instance_ = VK_NULL_HANDLE;
  uint32_t negotiatedApiVersion_ = 0;
  PFN_vkGetPhysicalDeviceFeatures2 physicalDeviceFeatures2Fn_ = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties2
      physicalDeviceMemoryProperties2Fn_ = nullptr;
  std::vector<VulkanDeviceInfo> devices_;

  /** Tracked allocation bytes per device (for budget approximation).
   *  std::atomic is non-moveable so we use a heap array instead of vector. */
  std::unique_ptr<std::atomic<uint64_t>[]> trackedBytes_;
  size_t trackedBytesCount_ = 0;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN

#endif  // LIBND4J_VULKAN_DEVICE_MANAGER_H
