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

#ifndef LIBND4J_HEXAGON_RUNTIME_MANAGER_H
#define LIBND4J_HEXAGON_RUNTIME_MANAGER_H

#include <system/common.h>

#ifdef HAVE_HEXAGON_MLIR

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * Singleton managing the hexagon-mlir compiler runtime.
 *
 * Loads libhexagon_mlir_runtime.so via dlopen at runtime and provides
 * access to NPU device management, TCM (Tightly Coupled Memory) allocation,
 * DDR allocation, kernel compilation from MLIR bytecode, and kernel dispatch.
 *
 * All hexagon types are opaque void* to avoid header dependencies on the
 * hexagon-mlir SDK. Function pointers are resolved via dlsym at load time.
 */
class SD_LIB_EXPORT HexagonRuntimeManager {
 public:
  /**
   * Get the singleton instance.
   * Thread-safe via std::call_once.
   */
  static HexagonRuntimeManager& getInstance();

  // Non-copyable, non-movable
  HexagonRuntimeManager(const HexagonRuntimeManager&) = delete;
  HexagonRuntimeManager& operator=(const HexagonRuntimeManager&) = delete;
  HexagonRuntimeManager(HexagonRuntimeManager&&) = delete;
  HexagonRuntimeManager& operator=(HexagonRuntimeManager&&) = delete;

  /**
   * Check if the hexagon-mlir runtime is loaded and an NPU device is present.
   */
  bool isAvailable() const;

  // ---- NPU Device Management ----

  /**
   * Get the number of available NPU devices.
   */
  int getDeviceCount() const;

  /**
   * Initialize an NPU device and return an opaque device context.
   * @param deviceId NPU device index (0-based)
   * @return Opaque NPU context handle, or nullptr on failure
   */
  void* initDevice(int deviceId = 0);

  /**
   * Release an NPU device context.
   * @param npuContext Context returned by initDevice()
   */
  void releaseDevice(void* npuContext);

  // ---- TCM (Tightly Coupled Memory) Management ----
  // TCM is on-chip SRAM (typically 256KB-1MB) with single-cycle access.
  // Used as staging memory for HVX vector operations.

  /**
   * Allocate TCM memory on the NPU.
   * @param npuContext NPU context from initDevice()
   * @param bytes Number of bytes to allocate (must fit in TCM capacity)
   * @return Opaque TCM pointer, or nullptr on failure
   */
  void* allocateTcm(void* npuContext, size_t bytes);

  /**
   * Free a TCM allocation.
   * @param npuContext NPU context
   * @param tcmPtr Pointer returned by allocateTcm()
   */
  void freeTcm(void* npuContext, void* tcmPtr);

  /**
   * Get the total TCM capacity for a device.
   * @param npuContext NPU context
   * @return TCM capacity in bytes, or 0 if unknown
   */
  size_t getTcmCapacity(void* npuContext) const;

  // ---- NPU DDR Memory Management ----
  // DDR accessible by the NPU for large buffers that don't fit in TCM.

  /**
   * Allocate NPU-accessible DDR memory.
   * @param npuContext NPU context
   * @param bytes Number of bytes to allocate
   * @return Opaque DDR pointer, or nullptr on failure
   */
  void* allocateDdr(void* npuContext, size_t bytes);

  /**
   * Free an NPU DDR allocation.
   * @param npuContext NPU context
   * @param ddrPtr Pointer returned by allocateDdr()
   */
  void freeDdr(void* npuContext, void* ddrPtr);

  // ---- Kernel Compilation ----

  /**
   * Compile MLIR bytecode into an NPU kernel.
   * @param npuContext NPU context
   * @param mlirBytecode MLIR bytecode buffer (Triton dialect targeting Hexagon)
   * @param bytecodeSize Size of the bytecode buffer
   * @return Opaque kernel handle, or nullptr on compilation failure
   */
  void* compileKernel(void* npuContext, const uint8_t* mlirBytecode,
                      size_t bytecodeSize);

  /**
   * Release a compiled kernel.
   * @param npuContext NPU context
   * @param kernelHandle Handle returned by compileKernel()
   */
  void releaseKernel(void* npuContext, void* kernelHandle);

  // ---- Kernel Dispatch ----

  /**
   * Dispatch a compiled kernel to the NPU.
   * @param npuContext NPU context
   * @param kernelHandle Compiled kernel from compileKernel()
   * @param args Array of argument pointers (TCM/DDR buffers)
   * @param numArgs Number of arguments
   * @return true if dispatch succeeded
   */
  bool dispatchKernel(void* npuContext, void* kernelHandle,
                      void** args, int numArgs);

  /**
   * Wait for all dispatched kernels to complete on the NPU.
   * @param npuContext NPU context
   * @return true if all operations completed successfully
   */
  bool waitForCompletion(void* npuContext);

  // ---- DMA Operations ----

  /**
   * Initiate a DMA transfer from host DDR to NPU TCM.
   * @param npuContext NPU context
   * @param dst TCM destination pointer
   * @param src Host source pointer
   * @param bytes Number of bytes to transfer
   * @return true if DMA was initiated successfully
   */
  bool dmaHostToTcm(void* npuContext, void* dst, const void* src, size_t bytes);

  /**
   * Initiate a DMA transfer from NPU TCM to host DDR.
   * @param npuContext NPU context
   * @param dst Host destination pointer
   * @param src TCM source pointer
   * @param bytes Number of bytes to transfer
   * @return true if DMA was initiated successfully
   */
  bool dmaTcmToHost(void* npuContext, void* dst, const void* src, size_t bytes);

  /**
   * Initiate a DMA transfer between DDR and TCM on the NPU.
   * @param npuContext NPU context
   * @param dst Destination pointer (TCM or DDR)
   * @param src Source pointer (TCM or DDR)
   * @param bytes Number of bytes to transfer
   * @return true if DMA was initiated successfully
   */
  bool dmaCopy(void* npuContext, void* dst, const void* src, size_t bytes);

 private:
  HexagonRuntimeManager();
  ~HexagonRuntimeManager();

  /**
   * Attempt to load the hexagon-mlir runtime library.
   * @return true if the library was loaded and all symbols resolved
   */
  bool loadRuntime();

  /**
   * Unload the runtime library and null all function pointers.
   */
  void unloadRuntime();

  // Runtime library handle
  void* libHandle_ = nullptr;
  bool available_ = false;
  mutable std::mutex mutex_;

  // ---- Function pointers resolved via dlsym ----
  // All follow the pattern: hexmlir_<category>_<action>

  // Device management
  using FnGetDeviceCount = int (*)();
  using FnInitDevice = void* (*)(int);
  using FnReleaseDevice = void (*)(void*);

  FnGetDeviceCount fnGetDeviceCount_ = nullptr;
  FnInitDevice fnInitDevice_ = nullptr;
  FnReleaseDevice fnReleaseDevice_ = nullptr;

  // TCM management
  using FnAllocateTcm = void* (*)(void*, size_t);
  using FnFreeTcm = void (*)(void*, void*);
  using FnGetTcmCapacity = size_t (*)(void*);

  FnAllocateTcm fnAllocateTcm_ = nullptr;
  FnFreeTcm fnFreeTcm_ = nullptr;
  FnGetTcmCapacity fnGetTcmCapacity_ = nullptr;

  // DDR management
  using FnAllocateDdr = void* (*)(void*, size_t);
  using FnFreeDdr = void (*)(void*, void*);

  FnAllocateDdr fnAllocateDdr_ = nullptr;
  FnFreeDdr fnFreeDdr_ = nullptr;

  // Kernel compilation and dispatch
  using FnCompileKernel = void* (*)(void*, const uint8_t*, size_t);
  using FnReleaseKernel = void (*)(void*, void*);
  using FnDispatchKernel = int (*)(void*, void*, void**, int);
  using FnWaitForCompletion = int (*)(void*);

  FnCompileKernel fnCompileKernel_ = nullptr;
  FnReleaseKernel fnReleaseKernel_ = nullptr;
  FnDispatchKernel fnDispatchKernel_ = nullptr;
  FnWaitForCompletion fnWaitForCompletion_ = nullptr;

  // DMA operations
  using FnDmaCopy = int (*)(void*, void*, const void*, size_t);

  FnDmaCopy fnDmaHostToTcm_ = nullptr;
  FnDmaCopy fnDmaTcmToHost_ = nullptr;
  FnDmaCopy fnDmaCopy_ = nullptr;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_HEXAGON_MLIR
#endif  // LIBND4J_HEXAGON_RUNTIME_MANAGER_H
