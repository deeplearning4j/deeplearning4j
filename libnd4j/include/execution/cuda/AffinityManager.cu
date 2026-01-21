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
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <execution/ContextBuffers.h>
#include <execution/LaunchContext.h>
#include <helpers/logger.h>

thread_local int globalThreadToDevice = -1;

// Defined in LaunchContext.cu - thread-local context buffers
extern thread_local sd::ContextBuffers contextBuffers;

namespace sd {

std::mutex AffinityManager::_currentMutex;
std::mutex AffinityManager::_numberMutex;
int AffinityManager::_numberOfDevices = -1;
std::vector<int> AffinityManager::_availableDevices;

/**
 * Check if the given device ID represents the CPU.
 * CPU is represented as device -1.
 */
bool isCpuDevice(int deviceId) {
  return deviceId == CPU_DEVICE_ID;
}

/**
 * Returns the CPU device ID constant.
 */
int getCpuDeviceId() {
  return CPU_DEVICE_ID;
}

int AffinityManager::currentDeviceId() {
  // CRITICAL FIX: Always query CUDA for the current device and sync our thread-local state.
  // This prevents race conditions where native code auto-assigns a device before Java
  // has a chance to set it via setDevice(). Java controls thread-device affinity through
  // CudaAffinityManager, and native code should respect whatever device Java has set.
  //
  // Previous behavior: Native code would auto-assign devices using its own round-robin,
  // which could conflict with Java's assignment when multiple threads start simultaneously.

  int dev = 0;
  auto res = cudaGetDevice(&dev);

  if (res != 0) throw cuda_exception::build("cudaGetDevice failed", res);

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

  if (res != 0) throw cuda_exception::build("cudaGetDevice failed", res);

  return dev;
}

int AffinityManager::numberOfDevices() {
  _numberMutex.lock();
  // we want to cache number of devices
  if (_numberOfDevices <= 0) {
    int dev = 0;
    auto res = cudaGetDeviceCount(&dev);

    if (res != 0) throw cuda_exception::build("cudaGetDeviceCount failed", res);

    _numberOfDevices = dev;
  }
  _numberMutex.unlock();

  return _numberOfDevices;
}

void AffinityManager::setCurrentNativeDevice(int deviceId) {
  auto res = cudaSetDevice(deviceId);
  if (res != 0) throw cuda_exception::build("setCurrentDevice failed", res);
}

void AffinityManager::setCurrentDevice(int deviceId) {
  auto previousDeviceId = globalThreadToDevice;

  // Check if context buffers need to be released due to device mismatch.
  // This handles two cases:
  // 1. Thread switching from one device to another (previousDeviceId >= 0 and different)
  // 2. NEW threads where context was lazily initialized for wrong device (previousDeviceId == -1)
  if (LaunchContext::isInitialized()) {
    int contextDeviceId = contextBuffers.deviceId();

    // Determine if we need to release and reinitialize
    bool needsRelease = false;
    if (previousDeviceId >= 0 && deviceId != previousDeviceId) {
      // Case 1: Explicit device switch
      needsRelease = true;
    } else if (contextDeviceId >= 0 && contextDeviceId != deviceId) {
      // Case 2: Context was initialized for wrong device
      needsRelease = true;
    }

    if (needsRelease) {
      // Release will handle device switching internally to properly sync streams
      LaunchContext::releaseBuffers();
    }
  }

  // Switch to target device
  auto res = cudaSetDevice(deviceId);
  if (res != 0) throw cuda_exception::build("cudaSetDevice failed", res);

  // update thread-device affinity
  globalThreadToDevice = deviceId;
}

void AffinityManager::syncThreadDeviceId(int deviceId) {
  // Sync the thread-local device ID with the specified device
  globalThreadToDevice = deviceId;
}

void AffinityManager::setAvailableDevices(const std::vector<int> &devices) {
  // For CUDA, this can be used to restrict which devices are available
  // Currently a no-op but could be extended to filter device selection
}

std::atomic<int> AffinityManager::_lastDevice;  // = std::atomic<int>(initialV);
}  // namespace sd
