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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace sd {
namespace graph {

// ============================================================
// PJRT/TPU Backend — STUB ONLY, not yet implemented.
//
// All methods in PjrtClientManager return false/nullptr/error.
// Do NOT call any of these methods in production code paths.
//
// The stub exists to allow the TPU execution path (GEM_TPU mode,
// TpuGraphBackend) to compile and link on non-TPU platforms without
// requiring libtpu.so or pjrt_c_api.h headers at build time.
//
// Production TPU support requires:
//   1. pjrt_c_api.h headers from XLA/libtpu SDK
//   2. libtpu.so available at runtime
//   3. Actual PJRT_Client_Create / PJRT_Client_Compile calls wired up
//      in loadLibrary() and initClient() below.
//
// See ADR 0072 for the TPU backend roadmap and integration plan.
// ============================================================

/**
 * Singleton managing PJRT_Api* function table and PJRT_Client lifecycle.
 *
 * Loads libtpu.so at runtime via dlopen, resolves the GetPjrtApi() symbol,
 * and provides a thread-safe interface for TPU device enumeration, buffer
 * management, and HLO compilation.
 *
 * All PJRT types are opaque void* since we dlopen at runtime and do not
 * link against any PJRT headers. The actual PJRT_Api struct is cast to
 * void* and function pointers are resolved from the symbol table.
 *
 * STUB STATUS: All methods currently return false/nullptr. See block comment
 * above for the implementation roadmap.
 */
class SD_LIB_EXPORT PjrtClientManager {
 public:
  /**
   * Get the singleton instance. Thread-safe lazy initialization.
   */
  static PjrtClientManager& getInstance();

  /**
   * Check if the PJRT library was loaded and client is available.
   * @return true if libtpu.so was loaded and a PJRT client was created
   */
  bool isAvailable() const;

  // ── Device enumeration ────────────────────────────────────────────────

  /**
   * Get the number of TPU devices visible to this client.
   * @return device count, or 0 if client is not initialized
   */
  int getDeviceCount() const;

  /**
   * Get opaque device handles for all visible TPU devices.
   * Each entry is a PJRT_Device* cast to void*.
   * @return vector of opaque device pointers
   */
  std::vector<void*> getDevices() const;

  // ── Buffer management ─────────────────────────────────────────────────

  /**
   * Create a device buffer from host data.
   *
   * @param hostData   Pointer to host memory
   * @param sizeBytes  Size of the data in bytes
   * @param deviceIdx  Target device index
   * @return Opaque PJRT_Buffer* handle, or nullptr on failure
   */
  void* createBuffer(const void* hostData, size_t sizeBytes, int deviceIdx = 0);

  /**
   * Destroy a previously created device buffer.
   *
   * @param buffer  Opaque PJRT_Buffer* handle from createBuffer()
   */
  void destroyBuffer(void* buffer);

  /**
   * Copy buffer contents back to host memory.
   *
   * @param buffer     Opaque PJRT_Buffer* handle
   * @param hostDst    Destination host memory (must be pre-allocated)
   * @param sizeBytes  Number of bytes to copy
   * @return true if the transfer completed successfully
   */
  bool bufferToHost(void* buffer, void* hostDst, size_t sizeBytes);

  // ── Compilation ───────────────────────────────────────────────────────

  /**
   * Compile an HLO module into an executable.
   *
   * @param hloBytes  Serialized HLO module (text or protobuf)
   * @param hloSize   Size of the HLO data in bytes
   * @return Opaque PJRT_LoadedExecutable* handle, or nullptr on failure
   */
  void* compile(const void* hloBytes, size_t hloSize);

  /**
   * Execute a compiled executable with input buffers.
   *
   * @param executable    Opaque PJRT_LoadedExecutable* from compile()
   * @param inputBuffers  Array of opaque PJRT_Buffer* input handles
   * @param numInputs     Number of input buffers
   * @param outputBuffers Output: array of opaque PJRT_Buffer* result handles
   * @param numOutputs    Output: number of result buffers
   * @return true if execution completed successfully
   */
  bool execute(void* executable, void** inputBuffers, int numInputs,
               void** outputBuffers, int* numOutputs);

  /**
   * Destroy a compiled executable.
   *
   * @param executable  Opaque PJRT_LoadedExecutable* from compile()
   */
  void destroyExecutable(void* executable);

  // ── Diagnostics ───────────────────────────────────────────────────────

  /**
   * Get the last error message from PJRT operations.
   */
  const std::string& getLastError() const { return lastError_; }

 private:
  PjrtClientManager();
  ~PjrtClientManager();

  // Non-copyable
  PjrtClientManager(const PjrtClientManager&) = delete;
  PjrtClientManager& operator=(const PjrtClientManager&) = delete;

  /**
   * Load libtpu.so and resolve GetPjrtApi symbol.
   * @return true if library was loaded successfully
   */
  bool loadLibrary();

  /**
   * Initialize the PJRT client from the loaded API.
   * @return true if client was created successfully
   */
  bool initClient();

  // Library handle from dlopen
  void* libHandle_ = nullptr;

  // PJRT_Api* function table (opaque — resolved at runtime)
  void* pjrtApi_ = nullptr;

  // PJRT_Client* (opaque)
  void* client_ = nullptr;

  // Cached device list
  std::vector<void*> devices_;

  // Thread safety for lazy init
  mutable std::mutex mutex_;
  bool initialized_ = false;
  bool initFailed_ = false;

  // Last error message
  std::string lastError_;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_PJRT_CLIENT_MANAGER_H
