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

#ifndef LIBND4J_PJRT_CLIENT_MANAGER_H
#define LIBND4J_PJRT_CLIENT_MANAGER_H

#include <system/common.h>

#ifdef SD_TPU

#include <array/NDArray.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * Process-wide owner of the runtime-loaded PJRT plugin and client.
 *
 * The PJRT C API header defines the ABI, but the plugin remains a runtime
 * dependency: libnd4jtpu never links libtpu (or another PJRT plugin). All
 * public handles are opaque so PJRT types do not leak through the graph API.
 */
class SD_LIB_EXPORT PjrtClientManager {
 public:
  /**
   * Process-lifetime singleton. Native plan shutdown deliberately leaves
   * backend runtimes alive until process exit, matching the plan-cache shutdown
   * contract and avoiding static-destruction order races with replay handles.
   */
  static PjrtClientManager& getInstance();

  /** Initialize the selected plugin and client. Safe to call repeatedly. */
  bool initialize();

  /** True when a client with at least one addressable device is ready. */
  bool isAvailable() const;

  /** True only when the loaded PJRT client reports a TPU platform. */
  bool isTpuPlatform() const;

  std::string getPlatformName() const;
  int getDeviceCount() const;
  std::vector<void*> getDevices() const;
  std::string getDeviceName(int deviceIdx) const;
  LongType getDeviceTotalMemory(int deviceIdx) const;
  LongType getDeviceFreeMemory(int deviceIdx) const;
  bool setCurrentDevice(int deviceIdx);
  int getCurrentDevice() const;

  /** Upload an NDArray value to one addressable PJRT device. */
  void* createBuffer(NDArray* array, int deviceIdx = 0);

  /** Destroy a PJRT_Buffer returned by createBuffer() or execute(). */
  void destroyBuffer(void* buffer);

  /** Copy one PJRT output buffer into a dense C-order NDArray. */
  bool bufferToArray(void* buffer, NDArray* destination);

  /** Compile an MLIR/StableHLO program for the selected device. */
  void* compile(const void* programBytes, size_t programSize,
                const char* programFormat = "mlir", int deviceIdx = 0);

  /**
   * Execute a loaded executable on one addressable device. Returned output
   * buffers are owned by the caller and must be destroyed with destroyBuffer().
   */
  bool execute(void* executable, void** inputBuffers, int numInputs,
               int deviceIdx, std::vector<void*>& outputBuffers);

  /** Release a loaded executable. */
  void destroyExecutable(void* executable);

  /** Monotonically invalidate backend compilation state. */
  void invalidateCompilationCache();
  uint64_t compilationGeneration() const;

  /** Thread-safe error snapshot. */
  std::string getLastError() const;

 private:
  PjrtClientManager();
  ~PjrtClientManager();

  PjrtClientManager(const PjrtClientManager&) = delete;
  PjrtClientManager& operator=(const PjrtClientManager&) = delete;

  static PjrtClientManager* instance_;
  static std::once_flag instanceOnce_;
  static thread_local int currentDevice_;

  bool loadLibrary();
  bool initClient();
  void shutdownUnlocked();
  bool consumeError(void* error, const char* operation);
  bool awaitAndDestroyEvent(void* event, const char* operation);
  void setLastError(const std::string& message);
  bool validDeviceIndex(int deviceIdx) const;

  void* libHandle_ = nullptr;
  void* pjrtApi_ = nullptr;
  void* client_ = nullptr;
  std::vector<void*> devices_;
  std::vector<std::string> deviceNames_;
  std::string platformName_;

  mutable std::mutex initMutex_;
  mutable std::mutex errorMutex_;
  mutable std::mutex executionMutex_;
  bool initialized_ = false;
  uint64_t compilationGeneration_ = 1;
  std::string lastError_;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_PJRT_CLIENT_MANAGER_H
