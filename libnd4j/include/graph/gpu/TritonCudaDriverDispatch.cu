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

//
// CUDA implementations for TritonCudaDriverDispatch.h.
// All CUDA Driver API and Runtime API calls used by Triton .cpp files live here.
// This file is compiled by NVCC only on CUDA builds (SD_CUDA defined).
//

#include <graph/gpu/TritonCudaDriverDispatch.h>

#ifdef SD_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>

namespace sd {
namespace graph {

// ─── Runtime API: device queries ───────────────────────────────────────────

int tritonGetCurrentDevice() {
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess) return -1;
  return dev;
}

int tritonGetDeviceCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
  return count;
}

const char* tritonCudaRtErrorString(int errorCode) {
  return cudaGetErrorString(static_cast<cudaError_t>(errorCode));
}

int tritonClearRtLastError() {
  return static_cast<int>(cudaGetLastError());
}

int tritonCudaDeviceGetAttribute(int* value, int attrib, int device) {
  return static_cast<int>(
      cudaDeviceGetAttribute(value, static_cast<cudaDeviceAttr>(attrib), device));
}

int tritonGetDeviceProperties(int device, char* name, int nameLen, int* smMajor, int* smMinor) {
  cudaDeviceProp props;
  cudaError_t err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) return static_cast<int>(err);
  if (name && nameLen > 0) {
    strncpy(name, props.name, static_cast<size_t>(nameLen - 1));
    name[nameLen - 1] = '\0';
  }
  if (smMajor) *smMajor = props.major;
  if (smMinor) *smMinor = props.minor;
  return 0;
}

// ─── Driver API: context management ────────────────────────────────────────

int tritonCtxGetCurrent(void** ctx) {
  CUcontext c = nullptr;
  CUresult res = cuCtxGetCurrent(&c);
  if (ctx) *ctx = static_cast<void*>(c);
  return static_cast<int>(res);
}

int tritonDeviceGet(void** dev, int ordinal) {
  // CUdevice is an int typedef — store it in a uintptr_t to avoid pointer truncation.
  // The caller holds it as void* but never dereferences it; it is passed back in
  // tritonDevicePrimaryCtxRetain / tritonDriverDeviceGetAttribute via the same cast.
  CUdevice d = 0;
  CUresult res = cuDeviceGet(&d, ordinal);
  if (dev) *dev = reinterpret_cast<void*>(static_cast<uintptr_t>(static_cast<unsigned int>(d)));
  return static_cast<int>(res);
}

int tritonDevicePrimaryCtxRetain(void** ctx, void* dev) {
  CUdevice d = static_cast<CUdevice>(
      static_cast<int>(reinterpret_cast<uintptr_t>(dev)));
  CUcontext c = nullptr;
  CUresult res = cuDevicePrimaryCtxRetain(&c, d);
  if (ctx) *ctx = static_cast<void*>(c);
  return static_cast<int>(res);
}

int tritonCtxPushCurrent(void* ctx) {
  return static_cast<int>(cuCtxPushCurrent(static_cast<CUcontext>(ctx)));
}

int tritonCtxPopCurrent(void** ctx) {
  CUcontext c = nullptr;
  CUresult res = cuCtxPopCurrent(&c);
  if (ctx) *ctx = static_cast<void*>(c);
  return static_cast<int>(res);
}

// ─── Driver API: module loading ─────────────────────────────────────────────

int tritonModuleLoadDataEx(void** module,
                            const void* ptxData,
                            char* jitErrorLog, int logBufSize,
                            char* jitInfoLog,
                            int generateLineInfo) {
  CUjit_option jitOptions[] = {
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_GENERATE_LINE_INFO
  };
  void* jitOptionValues[] = {
    jitErrorLog,
    reinterpret_cast<void*>(static_cast<uintptr_t>(logBufSize)),
    jitInfoLog,
    reinterpret_cast<void*>(static_cast<uintptr_t>(logBufSize)),
    reinterpret_cast<void*>(static_cast<uintptr_t>(generateLineInfo))
  };

  CUmodule mod = nullptr;
  CUresult res = cuModuleLoadDataEx(&mod, ptxData, 5, jitOptions, jitOptionValues);
  if (module) *module = static_cast<void*>(mod);
  return static_cast<int>(res);
}

int tritonGetDriverErrorString(int code, const char** errStr) {
  return static_cast<int>(cuGetErrorString(static_cast<CUresult>(code), errStr));
}

int tritonModuleGetFunction(void** func, void* module, const char* name) {
  CUfunction f = nullptr;
  CUresult res = cuModuleGetFunction(&f, static_cast<CUmodule>(module), name);
  if (func) *func = static_cast<void*>(f);
  return static_cast<int>(res);
}

int tritonModuleUnload(void* module) {
  if (!module) return 0;
  return static_cast<int>(cuModuleUnload(static_cast<CUmodule>(module)));
}

// ─── Driver API: kernel attributes ─────────────────────────────────────────

int tritonFuncSetAttribute(void* func, int attrib, int value) {
  return static_cast<int>(
      cuFuncSetAttribute(static_cast<CUfunction>(func),
                         static_cast<CUfunction_attribute>(attrib),
                         value));
}

// ─── Driver API: occupancy ──────────────────────────────────────────────────

int tritonDriverDeviceGetAttribute(int* value, int attrib, int deviceOrdinal) {
  CUdevice dev = 0;
  CUresult devRes = cuDeviceGet(&dev, deviceOrdinal);
  if (devRes != CUDA_SUCCESS) return static_cast<int>(devRes);
  return static_cast<int>(
      cuDeviceGetAttribute(value, static_cast<CUdevice_attribute>(attrib), dev));
}

int tritonOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                     void* func,
                                                     int blockSize,
                                                     size_t dynSharedMemPerBlock) {
  return static_cast<int>(
      cuOccupancyMaxActiveBlocksPerMultiprocessor(
          numBlocks,
          static_cast<CUfunction>(func),
          blockSize,
          dynSharedMemPerBlock));
}

// ─── Driver API: kernel launch ──────────────────────────────────────────────

int tritonLaunchKernel(void* func,
                        unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                        unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                        unsigned int sharedMemBytes,
                        void* stream,
                        void** args) {
  return static_cast<int>(
      cuLaunchKernel(static_cast<CUfunction>(func),
                     gridX, gridY, gridZ,
                     blockX, blockY, blockZ,
                     sharedMemBytes,
                     static_cast<CUstream>(stream),
                     args, nullptr));
}

int tritonLaunchCooperativeKernel(void* func,
                                   unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                   unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                   unsigned int sharedMemBytes,
                                   void* stream,
                                   void** args) {
  return static_cast<int>(
      cuLaunchCooperativeKernel(static_cast<CUfunction>(func),
                                 gridX, gridY, gridZ,
                                 blockX, blockY, blockZ,
                                 sharedMemBytes,
                                 static_cast<CUstream>(stream),
                                 args));
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
