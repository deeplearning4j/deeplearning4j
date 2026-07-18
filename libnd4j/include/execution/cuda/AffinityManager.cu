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
#include <string>
#include <execution/ContextBuffers.h>
#include <execution/LaunchContext.h>
#include <helpers/logger.h>

thread_local int globalThreadToDevice = -1;

// Defined in LaunchContext.cu - returns per-device ContextBuffers for current CUDA device
extern sd::ContextBuffers& contextBuffersForCurrentDevice();

namespace sd {

std::mutex AffinityManager::_currentMutex;
std::mutex AffinityManager::_numberMutex;
int AffinityManager::_numberOfDevices = -1;
std::vector<int> AffinityManager::_availableDevices;

/**
 * Check if the given device ID represents the CPU.
 * CPU is represented as device -1.
 */
bool SD_NS::isCpuDevice(int deviceId) {
  return deviceId == CPU_DEVICE_ID;
}

/**
 * Returns the CPU device ID constant.
 */
int SD_NS::getCpuDeviceId() {
  return CPU_DEVICE_ID;
}

int AffinityManager::currentDeviceId() {
  // Always query CUDA for the current device and sync our thread-local state.
  // This prevents race conditions where native code auto-assigns a device before Java
  // has a chance to set it via setDevice(). Java controls thread-device affinity through
  // CudaAffinityManager, and native code should respect whatever device Java has set.
  //
  // Previous behavior: Native code would auto-assign devices using its own round-robin,
  // which could conflict with Java's assignment when multiple threads start simultaneously.

  int dev = 0;
  auto res = cudaGetDevice(&dev);

  if (res != 0) {
    std::string msg = "cudaGetDevice failed; Error code: [" + std::to_string(res) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  // Sync thread-local cache with actual CUDA device
  // This ensures subsequent calls return the correct device without CUDA API overhead
  if (globalThreadToDevice != dev) {
    globalThreadToDevice = dev;
  }

  return dev;
}

int AffinityManager::currentNativeDeviceId() {
  int dev = 0;
  auto res = cudaGetDevice(&dev);

  if (res != 0) {
    std::string msg = "cudaGetDevice failed; Error code: [" + std::to_string(res) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  return dev;
}

int AffinityManager::numberOfDevices() {
  _numberMutex.lock();
  // we want to cache number of devices
  if (_numberOfDevices <= 0) {
    int dev = 0;
    auto res = cudaGetDeviceCount(&dev);

    if (res != 0) {
      std::string msg = "cudaGetDeviceCount failed; Error code: [" + std::to_string(res) + "]";
      THROW_EXCEPTION(msg.c_str());
    }

    _numberOfDevices = dev;
  }
  _numberMutex.unlock();

  return _numberOfDevices;
}

void AffinityManager::setCurrentNativeDevice(int deviceId) {
  auto res = cudaSetDevice(deviceId);
  if (res != 0) {
    std::string msg = "setCurrentDevice failed; Error code: [" + std::to_string(res) + "]";
    THROW_EXCEPTION(msg.c_str());
  }
}

void AffinityManager::setCurrentDevice(int deviceId) {
  auto previousDeviceId = globalThreadToDevice;

  // With per-device ContextBuffers map, no need to release buffers when switching
  // devices. Each device has its own persistent buffers (streams, workspace).
  // The per-device map in LaunchContext.cu handles routing automatically.

  // Switch to target device
  auto res = cudaSetDevice(deviceId);
  if (res != 0) {
    std::string msg = "cudaSetDevice failed; Error code: [" + std::to_string(res) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  // update thread-device affinity
  globalThreadToDevice = deviceId;
}

void AffinityManager::syncThreadDeviceId(int deviceId) {
  // Sync the thread-local device ID with the specified device
  globalThreadToDevice = deviceId;
}

void AffinityManager::setAvailableDevices(const std::vector<int> &devices) {
  std::lock_guard<std::mutex> lock(_currentMutex);
  _availableDevices = devices;

  if (_availableDevices.empty()) return;

  int currentDevice = -1;
  cudaError_t currentErr = cudaGetDevice(&currentDevice);
  bool currentAllowed = false;
  if (currentErr == cudaSuccess) {
    for (auto device : _availableDevices) {
      if (device == currentDevice) {
        currentAllowed = true;
        break;
      }
    }
  } else {
    cudaGetLastError();
  }

  if (currentAllowed) {
    globalThreadToDevice = currentDevice;
    return;
  }

  cudaError_t lastErr = cudaSuccess;
  for (auto device : _availableDevices) {
    lastErr = cudaSetDevice(device);
    if (lastErr == cudaSuccess) {
      globalThreadToDevice = device;
      cudaGetLastError();
      return;
    }

    sd_printf("AffinityManager: WARNING - cudaSetDevice(%d) failed while applying available devices: %s\n",
              device, cudaGetErrorString(lastErr));
    cudaGetLastError();
  }

  std::string msg = "setAvailableDevices failed to select any configured CUDA device; Error code: [" + std::to_string((int)lastErr) + "]";
  THROW_EXCEPTION(msg.c_str());
}

std::atomic<int> AffinityManager::_lastDevice;  // = std::atomic<int>(initialV);
}  // namespace sd
