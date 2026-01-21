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
#include <execution/AffinityManager.h>

namespace sd {

// Note: CPU_DEVICE_ID is defined in AffinityManager.h as -1 (interface constant)
// For internal CPU operations, we use 0 as the device ID
static constexpr int INTERNAL_CPU_DEVICE_ID = 0;

// Static member initialization for CPU backend
std::atomic<int> AffinityManager::_lastDevice{0};
int AffinityManager::_numberOfDevices = 1;
std::mutex AffinityManager::_currentMutex;
std::mutex AffinityManager::_numberMutex;
std::vector<int> AffinityManager::_availableDevices = {INTERNAL_CPU_DEVICE_ID};

// Thread-local device tracking - for CPU this is always 0 internally
thread_local int cpuThreadDevice = INTERNAL_CPU_DEVICE_ID;

// Implementation of isCpuDevice from header
bool isCpuDevice(int deviceId) {
  // Accept both the interface CPU_DEVICE_ID (-1) and internal device 0
  return deviceId == CPU_DEVICE_ID || deviceId == INTERNAL_CPU_DEVICE_ID;
}

// Implementation of getCpuDeviceId from header
int getCpuDeviceId() {
  return CPU_DEVICE_ID;
}

int AffinityManager::currentDeviceId() {
  // CPU is always internal device 0
  return INTERNAL_CPU_DEVICE_ID;
}

int AffinityManager::currentNativeDeviceId() {
  // CPU is always internal device 0
  return INTERNAL_CPU_DEVICE_ID;
}

int AffinityManager::numberOfDevices() {
  // CPU backend has exactly 1 device - the CPU itself
  return 1;
}

void AffinityManager::setCurrentDevice(int deviceId) {
  // For CPU, accept both CPU_DEVICE_ID (-1) and internal device 0
  // This is essentially a no-op since CPU has only one device, but we track the call
  cpuThreadDevice = INTERNAL_CPU_DEVICE_ID;
}

void AffinityManager::setCurrentNativeDevice(int deviceId) {
  // For CPU, only internal device 0 is valid
  cpuThreadDevice = INTERNAL_CPU_DEVICE_ID;
}

void AffinityManager::syncThreadDeviceId(int deviceId) {
  // For CPU - sync thread tracking to internal device 0
  cpuThreadDevice = INTERNAL_CPU_DEVICE_ID;
}

void AffinityManager::setAvailableDevices(const std::vector<int> &devices) {
  // For CPU, only internal device 0 is available
  // We ignore the input and keep the CPU device
  std::lock_guard<std::mutex> lock(_numberMutex);
  _availableDevices = {INTERNAL_CPU_DEVICE_ID};
  _numberOfDevices = 1;
}
}  // namespace sd
