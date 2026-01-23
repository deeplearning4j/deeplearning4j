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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 06.02.2019
// @author raver119@gmail.com
//
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/StringUtils.h>
#include <helpers/logger.h>
#include <memory/Workspace.h>
#include <memory/cuda/CudaMemoryPool.h>

#include "helpers/DebugHelper.h"

namespace sd {

//////////////////////////////////////////////////////////////////////////
PointersManager::PointersManager(const LaunchContext* context, const std::string& funcName) {
  _context = const_cast<LaunchContext*>(context);
  _funcName = funcName;
  _workspaceWasActive = (_context != nullptr && _context->getWorkspace() != nullptr);
}

//////////////////////////////////////////////////////////////////////////
void* PointersManager::allocateDevMem(const size_t sizeInBytes) {
  void* dst = nullptr;
  bool fromCudaMalloc = false;

  if (_context == nullptr || _context->getWorkspace() == nullptr) {
    // Use CUDA memory pool for efficient allocation
    auto& pool = memory::CudaMemoryPool::getInstance();
    int deviceId = 0;
    cudaGetDevice(&deviceId);

    // Get stream for async allocation if available
    cudaStream_t stream = nullptr;
    if (_context != nullptr && _context->getCudaStream() != nullptr) {
      stream = *_context->getCudaStream();
    }

    dst = pool.allocate(sizeInBytes, deviceId, stream);
    if (dst == nullptr)
      throw cuda_exception::build(_funcName + ": cannot allocate global memory on device!", cudaErrorMemoryAllocation);
    fromCudaMalloc = true;
  } else {
    // Allocate from workspace - workspace manages lifecycle
    dst = _context->getWorkspace()->allocateBytes(memory::MemoryType::DEVICE, sizeInBytes);
    fromCudaMalloc = false;
  }

  // Track allocation with its source
  _allocatedPointers.emplace_back(dst, fromCudaMalloc);
  return dst;
}

//////////////////////////////////////////////////////////////////////////
void* PointersManager::replicatePointer(const void* src, const size_t numberOfBytes) {
  // allocateDevMem already tracks the allocation
  void* dst = allocateDevMem(numberOfBytes);
  if (src) {
    if (_context != nullptr)
      cudaMemcpyAsync(dst, src, numberOfBytes, cudaMemcpyHostToDevice, *_context->getCudaStream());
    else
      cudaMemcpy(dst, src, numberOfBytes, cudaMemcpyHostToDevice);
  }
  // NOTE: We don't add to _allocatedPointers here because allocateDevMem already did

  return dst;
}

//////////////////////////////////////////////////////////////////////////
void PointersManager::synchronize() const {
  if (_context != nullptr) {
    cudaError_t cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
    if (cudaResult != 0) throw cuda_exception::build(_funcName + ": cuda stream synchronization failed !", cudaResult);
  } else {
    sd_printf("<%s> syncStream isn't possible: no stream set!", _funcName.c_str());
  }
}

//////////////////////////////////////////////////////////////////////////
PointersManager::~PointersManager() {
  if (_allocatedPointers.empty()) {
    return;
  }

  // Check if we have any cudaMalloc allocations that need freeing
  bool hasCudaMallocAllocations = false;
  for (const auto& alloc : _allocatedPointers) {
    if (alloc.fromCudaMalloc && alloc.ptr != nullptr) {
      hasCudaMallocAllocations = true;
      break;
    }
  }

  if (!hasCudaMallocAllocations) {
    // All allocations are from workspace - nothing to free
    return;
  }

  // Use CUDA memory pool for async free - no sync needed
  // cudaFreeAsync handles stream ordering automatically
  auto& pool = memory::CudaMemoryPool::getInstance();
  int deviceId = 0;
  cudaGetDevice(&deviceId);

  // Get stream for async free if available
  cudaStream_t stream = nullptr;
  if (_context != nullptr && _context->getCudaStream() != nullptr) {
    stream = *_context->getCudaStream();
  }

  // Free allocations via pool (returns memory to pool without blocking)
  for (const auto& alloc : _allocatedPointers) {
    if (alloc.fromCudaMalloc && alloc.ptr != nullptr) {
      pool.free(alloc.ptr, deviceId, stream);
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void printDevContentOnDev_(const void* pDev, const LongType len, const int tid) {
  PointersManager::printDevContentOnDev<T>(pDev, len, tid);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
void PointersManager::printDevContentOnDevFromHost(const void* pDev, const LongType len, const int tid) {
  printDevContentOnDev_<T><<<512, 512, 1024, *LaunchContext ::defaultContext()->getCudaStream()>>>(pDev, len, tid);
  auto res = cudaStreamSynchronize(*LaunchContext ::defaultContext()->getCudaStream());
  DebugHelper::checkGlobalErrorCode("concat general case failed(...) failed");

}
template void PointersManager::printDevContentOnDevFromHost<LongType>(const void* pDev, const LongType len,
                                                                          const int tid);
template void PointersManager::printDevContentOnDevFromHost<int>(const void* pDev, const LongType len,
                                                                 const int tid);
template void PointersManager::printDevContentOnDevFromHost<float>(const void* pDev, const LongType len,
                                                                   const int tid);
template void PointersManager::printDevContentOnDevFromHost<double>(const void* pDev, const LongType len,
                                                                    const int tid);


////////////////////////////////////////////////////////////////////////
template <typename T>
void PointersManager::printDevContentOnHost(const void* pDev, const LongType len) const {
  printf("host print out\n");
  void* pHost = operator new(sizeof(T) * len);

  cudaMemcpyAsync(pHost, pDev, sizeof(T) * len, cudaMemcpyDeviceToHost, *_context->getCudaStream());
  cudaError_t cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
  if (cudaResult != 0) THROW_EXCEPTION("PointersManager::printCudaHost: cudaStreamSynchronize failed!");

  for (LongType i = 0; i < len; ++i) printf("%f, ", (double)reinterpret_cast<T*>(pHost)[i]);
  printf("\n");

  operator delete(pHost);
}

template void PointersManager::printDevContentOnHost<LongType>(const void* pDev, const LongType len) const;
template void PointersManager::printDevContentOnHost<int>(const void* pDev, const LongType len) const;
template void PointersManager::printDevContentOnHost<float>(const void* pDev, const LongType len) const;
template void PointersManager::printDevContentOnHost<double>(const void* pDev, const LongType len) const;

}  // namespace sd
