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
// @author raver119@gmail.com
//

#ifndef LIBND4J_AFFINITYMANAGER_H
#define LIBND4J_AFFINITYMANAGER_H

#include <system/common.h>

#include <atomic>
#include <mutex>
#include <vector>

namespace sd {

/**
 * CPU device ID constant.
 * The CPU is acknowledged as device -1, enabling seamless CPU-GPU data transfer
 * in multi-backend scenarios. GPU devices use IDs >= 0.
 */
static constexpr int CPU_DEVICE_ID = -1;

/**
 * Check if the given device ID represents the CPU.
 * @param deviceId the device ID to check
 * @return true if deviceId == CPU_DEVICE_ID (-1), false otherwise
 */
SD_LIB_EXPORT bool isCpuDevice(int deviceId);

/**
 * Returns the CPU device ID constant (-1).
 * @return CPU_DEVICE_ID
 */
SD_LIB_EXPORT int getCpuDeviceId();

/**
 * Device type enumeration for distinguishing between CPU and GPU devices.
 */
enum class DeviceType {
  CPU,
  GPU,
  UNKNOWN
};

/**
 * Returns the device type for the given device ID.
 * @param deviceId the device ID to query
 * @return CPU for deviceId == -1, GPU for deviceId >= 0, UNKNOWN otherwise
 */
inline DeviceType getDeviceType(int deviceId) {
  if (deviceId == CPU_DEVICE_ID) return DeviceType::CPU;
  if (deviceId >= 0) return DeviceType::GPU;
  return DeviceType::UNKNOWN;
}

class SD_LIB_EXPORT AffinityManager {
 private:
  static std::atomic<int> _lastDevice;
  static int _numberOfDevices;
  static std::mutex _currentMutex;
  static std::mutex _numberMutex;
  static std::vector<int> _availableDevices;

 public:
  static int currentNativeDeviceId();
  static int currentDeviceId();
  static int numberOfDevices();
  static void setCurrentDevice(int deviceId);
  static void setCurrentNativeDevice(int deviceId);
  static void setAvailableDevices(const std::vector<int> &devices);

  /**
   * Lightweight method to sync thread-local device tracking (globalThreadToDevice)
   * with the actual CUDA device without expensive stream synchronization.
   * Use this when you need to ensure tracking consistency but don't need stream sync.
   */
  static void syncThreadDeviceId(int deviceId);

  /**
   * Returns the CPU device ID constant (-1).
   * CPU is always available as a target device for data placement.
   */
  static inline int cpuDeviceId() { return CPU_DEVICE_ID; }

  /**
   * Check if the given device ID represents the CPU.
   */
  static inline bool isCpu(int deviceId) { return deviceId == CPU_DEVICE_ID; }
};
}  // namespace sd

#endif  // DEV_TESTS_AFFINITYMANAGER_H
