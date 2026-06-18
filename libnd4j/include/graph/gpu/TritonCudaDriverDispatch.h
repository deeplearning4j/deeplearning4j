/* ******************************************************************************
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

#pragma once

//
// Unified dispatch header for CUDA Driver API and Runtime API calls used
// by Triton .cpp files.  CUDA implementation lives in TritonCudaDriverDispatch.cu.
// CPU builds get inline stubs — no #ifdef SD_CUDA needed at call sites.
//
// All CUDA Driver API types (CUmodule, CUfunction, CUresult, CUcontext,
// CUdevice, etc.) are hidden behind void* / int throughout this interface.
//

#include <cstddef>
#include <cstdint>

namespace sd {
namespace graph {

// ─── CUDA error codes returned as int ──────────────────────────────────────
//
// CUDA_SUCCESS == 0 for both Runtime (cudaSuccess) and Driver API.
// All triton* functions return 0 on success, non-zero on failure.
// This is consistent with both cudaError_t and CUresult.

static constexpr int TRITON_CUDA_SUCCESS = 0;

// ─── CUdevice_attribute constants (from cuda.h) ─────────────────────────────
// Named aliases so .cpp files never need to include <cuda.h>.

static constexpr int TRITON_CU_DEV_ATTR_MULTIPROCESSOR_COUNT              = 16;
static constexpr int TRITON_CU_DEV_ATTR_MAX_THREADS_PER_MULTIPROCESSOR    = 39;
static constexpr int TRITON_CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81;
static constexpr int TRITON_CU_DEV_ATTR_COOPERATIVE_LAUNCH                = 95;

// ─── CUfunction_attribute constants (from cuda.h) ───────────────────────────

static constexpr int TRITON_CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES    = 8;

// ─── Runtime API: device queries ───────────────────────────────────────────

/** cudaGetDevice. Returns device id; -1 on failure. */
int tritonGetCurrentDevice();

/** cudaGetDeviceCount. Returns count; 0 on failure. */
int tritonGetDeviceCount();

/** cudaGetErrorString for a cudaError_t code. */
const char* tritonCudaRtErrorString(int errorCode);

/** cudaGetLastError — clears sticky runtime error. Returns old error code. */
int tritonClearRtLastError();

/** cudaDeviceGetAttribute wrapper. attrib is int (cudaDeviceAttr enum value).
 *  Returns 0 (cudaSuccess) on success. */
int tritonCudaDeviceGetAttribute(int* value, int attrib, int device);

/** cudaGetDeviceProperties: fill name (max nameLen chars) and SM version (major*10+minor).
 *  Returns 0 on success. name may be nullptr. smVersion may be nullptr. */
int tritonGetDeviceProperties(int device, char* name, int nameLen, int* smMajor, int* smMinor);

// ─── Driver API: context management ────────────────────────────────────────

/** cuCtxGetCurrent. Returns 0 on success; *ctx is void* (CUcontext). */
int tritonCtxGetCurrent(void** ctx);

/** cuDeviceGet. Returns 0 on success; *dev is void* opaque handle (CUdevice stored as uintptr). */
int tritonDeviceGet(void** dev, int ordinal);

/** cuDevicePrimaryCtxRetain. Returns 0 on success; *ctx is void* (CUcontext). */
int tritonDevicePrimaryCtxRetain(void** ctx, void* dev);

/** cuCtxPushCurrent. Returns 0 on success. */
int tritonCtxPushCurrent(void* ctx);

/** cuCtxPopCurrent. Returns 0 on success; *ctx receives the popped context. */
int tritonCtxPopCurrent(void** ctx);

// ─── Driver API: module loading ─────────────────────────────────────────────

/** cuModuleLoadDataEx with JIT error/info log buffers.
 *  On success, *module is a non-null void* (CUmodule).
 *  jitErrorLog / jitInfoLog must be pre-allocated buffers of logBufSize bytes.
 *  generateLineInfo: 0=disable, 1=enable.
 *  Returns 0 (CUDA_SUCCESS) on success. */
int tritonModuleLoadDataEx(void** module,
                            const void* ptxData,
                            char* jitErrorLog, int logBufSize,
                            char* jitInfoLog,
                            int generateLineInfo);

/** cuGetErrorString for a CUresult code. errStr is set to the description.
 *  Returns 0 on success. */
int tritonGetDriverErrorString(int code, const char** errStr);

/** cuModuleGetFunction. Returns 0 on success; *func is non-null void* (CUfunction). */
int tritonModuleGetFunction(void** func, void* module, const char* name);

/** cuModuleUnload. Always returns 0 (errors are non-fatal). */
int tritonModuleUnload(void* module);

// ─── Driver API: kernel attributes ─────────────────────────────────────────

/** cuFuncSetAttribute. attrib is int (CUfunction_attribute enum value).
 *  Returns 0 on success. */
int tritonFuncSetAttribute(void* func, int attrib, int value);

// ─── Driver API: occupancy ──────────────────────────────────────────────────

/** cuDeviceGetAttribute. attrib is int (CUdevice_attribute enum value).
 *  device is the integer device ordinal (NOT a CUdevice handle).
 *  Returns 0 on success. */
int tritonDriverDeviceGetAttribute(int* value, int attrib, int deviceOrdinal);

/** cuOccupancyMaxActiveBlocksPerMultiprocessor.
 *  func is void* (CUfunction). Returns 0 on success. */
int tritonOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                     void* func,
                                                     int blockSize,
                                                     size_t dynSharedMemPerBlock);

// ─── Driver API: kernel launch ──────────────────────────────────────────────

/** cuLaunchKernel. stream is void* (CUstream). Returns 0 on success. */
int tritonLaunchKernel(void* func,
                        unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                        unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                        unsigned int sharedMemBytes,
                        void* stream,
                        void** args);

/** cuLaunchCooperativeKernel. stream is void* (CUstream). Returns 0 on success. */
int tritonLaunchCooperativeKernel(void* func,
                                   unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                   unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                   unsigned int sharedMemBytes,
                                   void* stream,
                                   void** args);

}  // namespace graph
}  // namespace sd

#ifndef SD_CUDA

// ─── CPU inline stubs ────────────────────────────────────────────────────────
// These make Triton .cpp files compile and link on CPU without any CUDA headers.

namespace sd {
namespace graph {

static inline int  tritonGetCurrentDevice() { return -1; }
static inline int  tritonGetDeviceCount()   { return 0; }
static inline const char* tritonCudaRtErrorString(int) { return "n/a (CPU)"; }
static inline int  tritonClearRtLastError() { return 0; }
static inline int  tritonCudaDeviceGetAttribute(int* v, int, int) { if (v) *v = 0; return -1; }
static inline int  tritonGetDeviceProperties(int, char*, int, int* maj, int* min) {
  if (maj) *maj = 0; if (min) *min = 0; return -1;
}

static inline int  tritonCtxGetCurrent(void** ctx)                        { if (ctx) *ctx = nullptr; return -1; }
static inline int  tritonDeviceGet(void** dev, int)                        { if (dev) *dev = nullptr; return -1; }
static inline int  tritonDevicePrimaryCtxRetain(void** ctx, void*)         { if (ctx) *ctx = nullptr; return -1; }
static inline int  tritonCtxPushCurrent(void*)                             { return -1; }
static inline int  tritonCtxPopCurrent(void** ctx)                         { if (ctx) *ctx = nullptr; return -1; }

static inline int  tritonModuleLoadDataEx(void** m, const void*, char*, int, char*, int) {
  if (m) *m = nullptr; return -1;
}
static inline int  tritonGetDriverErrorString(int, const char** s) { if (s) *s = "n/a (CPU)"; return 0; }
static inline int  tritonModuleGetFunction(void** f, void*, const char*) { if (f) *f = nullptr; return -1; }
static inline int  tritonModuleUnload(void*) { return 0; }

static inline int  tritonFuncSetAttribute(void*, int, int) { return -1; }
static inline int  tritonDriverDeviceGetAttribute(int* v, int, int) { if (v) *v = 0; return -1; }
static inline int  tritonOccupancyMaxActiveBlocksPerMultiprocessor(int* b, void*, int, size_t) {
  if (b) *b = 0; return -1;
}

static inline int  tritonLaunchKernel(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                       unsigned, void*, void**) { return -1; }
static inline int  tritonLaunchCooperativeKernel(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                                  unsigned, void*, void**) { return -1; }

}  // namespace graph
}  // namespace sd

#endif  // !SD_CUDA
