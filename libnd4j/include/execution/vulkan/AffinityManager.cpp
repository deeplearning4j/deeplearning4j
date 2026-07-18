/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <execution/AffinityManager.h>
#include <graph/vulkan/VulkanDeviceManager.h>

#include <algorithm>
#include <string>

namespace sd {

std::atomic<int> AffinityManager::_lastDevice{0};
int AffinityManager::_numberOfDevices = -1;
std::mutex AffinityManager::_currentMutex;
std::mutex AffinityManager::_numberMutex;
std::vector<int> AffinityManager::_availableDevices;

bool SD_NS::isCpuDevice(int deviceId) {
  return deviceId == CPU_DEVICE_ID;
}

int SD_NS::getCpuDeviceId() {
  return CPU_DEVICE_ID;
}

namespace {

graph::VulkanDeviceManager& initializedDeviceManager() {
  auto& manager = graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    THROW_EXCEPTION("Vulkan device initialization failed");
  }
  if (manager.deviceCount() <= 0) {
    THROW_EXCEPTION("No Vulkan devices are available");
  }
  return manager;
}

void validateDeviceId(int deviceId, int deviceCount) {
  if (deviceId < 0 || deviceId >= deviceCount) {
    std::string message = "Invalid Vulkan device id " + std::to_string(deviceId) +
                          "; enumerated device count is " + std::to_string(deviceCount);
    THROW_EXCEPTION(message.c_str());
  }
}

}  // namespace

int AffinityManager::currentDeviceId() {
  auto& manager = initializedDeviceManager();
  const int deviceId = graph::VulkanDeviceManager::currentDeviceId();
  validateDeviceId(deviceId, manager.deviceCount());
  return deviceId;
}

int AffinityManager::currentNativeDeviceId() {
  return currentDeviceId();
}

int AffinityManager::numberOfDevices() {
  std::lock_guard<std::mutex> lock(_numberMutex);
  auto& manager = initializedDeviceManager();
  _numberOfDevices = manager.deviceCount();
  return _numberOfDevices;
}

void AffinityManager::setCurrentNativeDevice(int deviceId) {
  setCurrentDevice(deviceId);
}

void AffinityManager::setCurrentDevice(int deviceId) {
  auto& manager = initializedDeviceManager();
  validateDeviceId(deviceId, manager.deviceCount());
  if (!manager.setCurrentDevice(deviceId)) {
    std::string message = "Failed to select Vulkan device " + std::to_string(deviceId);
    THROW_EXCEPTION(message.c_str());
  }
}

void AffinityManager::syncThreadDeviceId(int deviceId) {
  setCurrentDevice(deviceId);
}

void AffinityManager::setAvailableDevices(const std::vector<int>& devices) {
  auto& manager = initializedDeviceManager();
  const int deviceCount = manager.deviceCount();

  std::lock_guard<std::mutex> lock(_currentMutex);
  for (int deviceId : devices) {
    validateDeviceId(deviceId, deviceCount);
  }
  _availableDevices = devices;

  // An empty scheduler allow-list means all enumerated Vulkan devices. It does
  // not alter loader visibility or the physical-device enumeration.
  if (_availableDevices.empty()) {
    return;
  }

  const int current = graph::VulkanDeviceManager::currentDeviceId();
  if (std::find(_availableDevices.begin(), _availableDevices.end(), current) !=
      _availableDevices.end()) {
    return;
  }

  if (!manager.setCurrentDevice(_availableDevices.front())) {
    std::string message =
        "Failed to select configured Vulkan device " + std::to_string(_availableDevices.front());
    THROW_EXCEPTION(message.c_str());
  }
}

}  // namespace sd
