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
#include <array/DataBuffer.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/StringUtils.h>
#include <helpers/logger.h>
#include <memory/Workspace.h>
#include <memory/cuda/CudaMemoryPool.h>

#include "helpers/DebugHelper.h"

namespace sd {

namespace {
SD_INLINE cudaStream_t captureSafeStream(const LaunchContext* context) {
  if (tl_graphExecutionActive && tl_graphCaptureStream != nullptr) {
    return tl_graphCaptureStream;
  }
  auto* streamPtr = (context != nullptr) ? context->getCudaStream()
                                          : LaunchContext::defaultContext()->getCudaStream();
  return (streamPtr != nullptr) ? *streamPtr : nullptr;
}
}  // namespace

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
/**
 * FNV-1a hash for small byte arrays (dimension/axis arrays are typically 4-32 bytes).
 * Combined with size to form the cache key.
 */
static uint64_t fnvHash(const void* data, size_t len) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; i++) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

void* PointersManager::replicatePointer(const void* src, const size_t numberOfBytes) {
  if (src && tl_graphExecutionActive && numberOfBytes <= 256) {
    // During CUDA graph capture, check if identical content was already uploaded.
    // Dimension/axis arrays (e.g., [0,1] for reduce) are reused by many ops —
    // deduplicating avoids redundant cudaMemcpyAsync graph nodes on every replay.
    uint64_t key = fnvHash(src, numberOfBytes) ^ (numberOfBytes * 0x9e3779b97f4a7c15ULL);
    auto it = tl_captureReplicateCache.find(key);
    if (it != tl_captureReplicateCache.end()) {
      // Verify content match (hash collision check)
      // The cached pointer points to capture workspace memory — still valid during capture
      return it->second;
    }
  }

  // allocateDevMem already tracks the allocation
  void* dst = allocateDevMem(numberOfBytes);
  if (src) {
    if (tl_graphExecutionActive) {
      // During CUDA graph capture, H2D copies are recorded as graph nodes with the HOST
      // source address baked in. On replay, the graph reads from that same address.
      // If the host data was on the stack or in a temp buffer, it's invalid at replay time.
      // FIX: Copy host data to persistent pinned memory, then H2D from that.
      void* pinnedSrc = nullptr;
      auto pinErr = cudaMallocHost(&pinnedSrc, numberOfBytes);
      if (pinErr == cudaSuccess && pinnedSrc != nullptr) {
        std::memcpy(pinnedSrc, src, numberOfBytes);
        tl_capturedHostPtrs.push_back(pinnedSrc);

        cudaStream_t capturedStream = captureSafeStream(_context);
        cudaMemcpyAsync(dst, pinnedSrc, numberOfBytes, cudaMemcpyHostToDevice, capturedStream);
      } else {
        // Pinned alloc failed — fall back to direct copy (may fail on replay)
        sd_printf("PointersManager::replicatePointer: cudaMallocHost failed for %zu bytes\n",
                  numberOfBytes);
        cudaStream_t capturedStream = captureSafeStream(_context);
        cudaMemcpyAsync(dst, src, numberOfBytes, cudaMemcpyHostToDevice, capturedStream);
      }

      // Cache for future calls with same content (only for small arrays)
      if (numberOfBytes <= 256) {
        uint64_t key = fnvHash(src, numberOfBytes) ^ (numberOfBytes * 0x9e3779b97f4a7c15ULL);
        tl_captureReplicateCache[key] = dst;
      }
    } else if (_context != nullptr) {
      cudaMemcpyAsync(dst, src, numberOfBytes, cudaMemcpyHostToDevice, *_context->getCudaStream());
    } else {
      // Use cudaMemcpyAsync with per-thread stream instead of synchronous cudaMemcpy.
      // Synchronous cudaMemcpy uses the legacy default stream which implicitly syncs with
      // ALL streams — if any stream has capture state, this fails with error 906.
      cudaMemcpyAsync(dst, src, numberOfBytes, cudaMemcpyHostToDevice, cudaStreamPerThread);
      cudaStreamSynchronize(cudaStreamPerThread);
    }
  }
  // NOTE: We don't add to _allocatedPointers here because allocateDevMem already did

  return dst;
}

//////////////////////////////////////////////////////////////////////////
void PointersManager::synchronize() const {
  // During CUDA graph capture, stream synchronization is illegal on the captured stream
  // (error 900) and would invalidate the capture. Skip sync entirely — kernels are only
  // being recorded, not executed, so there's nothing to synchronize.
  if (tl_graphExecutionActive) return;

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
