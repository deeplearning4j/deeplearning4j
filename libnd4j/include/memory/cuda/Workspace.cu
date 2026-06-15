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
// CUDA workspaces implementation
//
// @author raver119@gmail.com
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <exceptions/cuda_exception.h>
#include <helpers/logger.h>
#include <math/templatemath.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <stdio.h>
#include <stdlib.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>

#include <execution/LaunchContext.h>

#include <atomic>
#include <cstring>

#include "../Workspace.h"

namespace sd {
namespace memory {

// Helper: get the current CUDA compute stream (never nullptr unless no context).
static cudaStream_t currentStream() {
  auto ctx = LaunchContext::defaultContext();
  if (ctx != nullptr && ctx->getCudaStream() != nullptr)
    return *ctx->getCudaStream();
  return nullptr;
}
Workspace::Workspace(ExternalWorkspace *external) {
  if (external->sizeHost() > 0) {
    _ptrHost = (char *)external->pointerHost();
    _ptrDevice = (char *)external->pointerDevice();

    _initialSize = external->sizeDevice();
    _currentSize = external->sizeDevice();
    _initialSizeSecondary = external->sizeHost();
    _currentSizeSecondary = external->sizeHost();
    _offset = 0L;
    _offsetSecondary = 0L;
    this->_cycleAllocations = 0;
    this->_cycleAllocationsSecondary = 0;
    this->_spillsSize = 0;
    this->_spillsSizeSecondary = 0;

    _externalized = true;
  }
}

Workspace::Workspace(LongType primarySize, LongType secondarySize) {
  if (secondarySize > 0) {
    // Over-allocate by CANARY_SIZE so enableCanary() can write a sentinel region
    // after the usable host buffer.  The canary is only checked in debug mode.
    auto res = cudaHostAlloc(reinterpret_cast<void **>(&_ptrHost),
                             secondarySize + CANARY_SIZE, cudaHostAllocDefault);
    if (res != 0) throw cuda_exception::build("Can't allocate [HOST] memory", res);

    // Host memory from cudaHostAlloc is CPU-accessible, use memset directly
    std::memset(this->_ptrHost, 0, secondarySize);
    // Pre-fill canary region and enable by default (only checked in debug+verbose)
    std::memset(this->_ptrHost + secondarySize, CANARY_BYTE, CANARY_SIZE);
    _canaryEnabled = true;
    this->_allocatedHost = true;
  } else
    this->_allocatedHost = false;

  if (primarySize > 0) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    this->_deviceId = deviceId;  // Store device ID for proper deallocation
    cudaStream_t stream = currentStream();
    _ptrDevice = reinterpret_cast<char*>(CudaMemoryPool::getInstance().allocate(primarySize, deviceId, stream));
    if (_ptrDevice == nullptr) throw cuda_exception::build("Can't allocate [DEVICE] memory", cudaErrorMemoryAllocation);

    // Use cudaMemsetAsync on the SAME stream as the allocation to maintain correct
    // stream ordering. cudaMallocAsync makes memory available on its stream, so
    // cudaMemsetAsync on the same stream is guaranteed to execute after the allocation.
    // Using synchronous cudaMemset (which runs on stream 0) would require implicit
    // synchronization via legacy default stream semantics, which breaks under
    // per-thread default stream mode.
    cudaMemsetAsync(this->_ptrDevice, 0, primarySize, stream);
    this->_allocatedDevice = true;
  } else {
    this->_allocatedDevice = false;
    this->_deviceId = -1;
  }

  this->_initialSize = primarySize;
  this->_initialSizeSecondary = secondarySize;
  this->_currentSize = primarySize;
  this->_currentSizeSecondary = secondarySize;
  this->_offset = 0;
  this->_offsetSecondary = 0;
  this->_cycleAllocations = 0;
  this->_spillsSize = 0;
  this->_spillsSizeSecondary = 0;
}

void Workspace::init(LongType primaryBytes, LongType secondaryBytes) {
  if (this->_currentSize < primaryBytes) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaStream_t stream = currentStream();
    if (this->_allocatedDevice && !_externalized) {
      // Use stored device ID if available, otherwise use current device
      int freeDeviceId = (this->_deviceId >= 0) ? this->_deviceId : deviceId;
      CudaMemoryPool::getInstance().free((void *)this->_ptrDevice, freeDeviceId, stream);
    }

    _ptrDevice = reinterpret_cast<char*>(CudaMemoryPool::getInstance().allocate(primaryBytes, deviceId, stream));
    if (_ptrDevice == nullptr) throw cuda_exception::build("Can't allocate [DEVICE] memory", cudaErrorMemoryAllocation);

    // Store device ID for proper deallocation later
    this->_deviceId = deviceId;

    // Use same stream as allocation for correct stream ordering
    cudaMemsetAsync(this->_ptrDevice, 0, primaryBytes, stream);
    this->_currentSize = primaryBytes;
    this->_allocatedDevice = true;
  }

  if (this->_currentSizeSecondary < secondaryBytes) {
    if (this->_allocatedHost && !_externalized) cudaFreeHost((void *)this->_ptrHost);

    auto res = cudaHostAlloc(reinterpret_cast<void **>(&_ptrHost),
                             secondaryBytes + CANARY_SIZE, cudaHostAllocDefault);
    if (res != 0) throw cuda_exception::build("Can't allocate [HOST] memory", res);

    // Host memory from cudaHostAlloc is CPU-accessible, use memset directly
    std::memset(this->_ptrHost, 0, secondaryBytes);
    // Re-fill canary after the new buffer
    std::memset(this->_ptrHost + secondaryBytes, CANARY_BYTE, CANARY_SIZE);
    this->_currentSizeSecondary = secondaryBytes;
    this->_allocatedHost = true;
    _canaryEnabled = true;
  }
}

void Workspace::expandBy(LongType numBytes, LongType secondaryBytes) {
  this->init(_currentSize + numBytes, _currentSizeSecondary + secondaryBytes);
}

void Workspace::expandTo(LongType numBytes, LongType secondaryBytes) { this->init(numBytes, secondaryBytes); }

void Workspace::freeSpills() {
  _spillsSize = 0;
  _spillsSizeSecondary = 0;

  // Only fetch device/stream if there are actual spills to free.
  // currentStream() triggers ContextBuffers::initialize() which creates CUDA streams.
  // When called from scopeIn() on a fresh workspace (no spills), this unnecessary
  // initialization can fail with cudaErrorMemoryAllocation if GPU memory is tight
  // (e.g., after a large DSP execution that hasn't fully released pool memory).
  if (!_spills.empty()) {
    int deviceId = (this->_deviceId >= 0) ? this->_deviceId : 0;
    if (this->_deviceId < 0) {
      cudaGetDevice(&deviceId);
    }
    cudaStream_t stream = currentStream();
    for (auto v : _spills) {
      CudaMemoryPool::getInstance().free(v, deviceId, stream);
    }
    _spills.clear();
  }

  if (!_spillsSecondary.empty()) {
    for (auto v : _spillsSecondary) cudaFreeHost(v);
    _spillsSecondary.clear();
  }
}

Workspace::~Workspace() {
  if (this->_allocatedHost && !_externalized) cudaFreeHost((void *)this->_ptrHost);

  if (this->_allocatedDevice && !_externalized) {
    // Use stored device ID if available, otherwise fall back to current device
    int deviceId = (this->_deviceId >= 0) ? this->_deviceId : 0;
    if (this->_deviceId < 0) {
      // Fallback: get current device if stored ID is not available
      cudaGetDevice(&deviceId);
    }
    CudaMemoryPool::getInstance().free((void *)this->_ptrDevice, deviceId, currentStream());
  }

  freeSpills();
}

LongType Workspace::getUsedSize() { return getCurrentOffset(); }

LongType Workspace::getCurrentSize() { return _currentSize; }

LongType Workspace::getCurrentOffset() { return _offset.load(); }

void *Workspace::allocateBytes(LongType numBytes) { return allocateBytes(HOST, numBytes); }

LongType Workspace::getAllocatedSize() { return getCurrentSize() + getSpilledSize(); }

void Workspace::scopeIn() {
  freeSpills();
  init(_cycleAllocations.load(), _cycleAllocationsSecondary.load());
  _cycleAllocations = 0;
  _cycleAllocationsSecondary = 0;
}

void Workspace::scopeOut() {
  // In debug mode, check canary before resetting offsets.
  // If canary is corrupted, the last op wrote past the workspace boundary.
  if (_canaryEnabled && sd::Environment::getInstance().isDebugAndVerbose()) {
    LongType corruptedAt = checkCanary();
    if (corruptedAt >= 0) {
      sd_printf("WORKSPACE CANARY CORRUPTED at offset %lld (workspace size %lld, host ptr %p). "
                "Last op wrote past workspace boundary!\n",
                corruptedAt, _currentSizeSecondary, _ptrHost);
      // Re-fill canary so we can detect the NEXT corruption too
      std::memset(_ptrHost + _currentSizeSecondary, CANARY_BYTE, CANARY_SIZE);
    }
  }
  _offset = 0;
  _offsetSecondary = 0;
}

void Workspace::enableCanary() {
  if (_allocatedHost && _ptrHost != nullptr) {
    std::memset(_ptrHost + _currentSizeSecondary, CANARY_BYTE, CANARY_SIZE);
    _canaryEnabled = true;
  }
}

LongType Workspace::checkCanary() const {
  if (!_canaryEnabled || !_allocatedHost || _ptrHost == nullptr) return -1;
  const unsigned char* canary = reinterpret_cast<const unsigned char*>(
      _ptrHost + _currentSizeSecondary);
  for (LongType i = 0; i < CANARY_SIZE; i++) {
    if (canary[i] != CANARY_BYTE) return i;
  }
  return -1;
}

LongType Workspace::getSpilledSize() { return _spillsSize.load(); }

void *Workspace::allocateBytes(MemoryType type, LongType numBytes) {
  switch (type) {
    case HOST: {
      if (numBytes < 1)
        throw allocation_exception::build("Number of [HOST] bytes for allocation should be positive", numBytes);

      // numBytes += 32;
      void *result = nullptr;
      this->_cycleAllocationsSecondary += numBytes;
      this->_mutexAllocation.lock();

      if (_offsetSecondary.load() + numBytes > _currentSizeSecondary) {
        sd_debug("Allocating %lld [HOST] bytes in spills\n", numBytes);
        this->_mutexAllocation.unlock();

        // Add padding to spill allocations — C++ ops can overrun temporary buffers
        // by a few bytes, corrupting adjacent heap metadata → SIGABRT on free().
        // Within the workspace buffer, overruns are harmless (bump allocator).
        // Spills go to separate cudaHostAlloc allocations where overruns hit
        // adjacent glibc/pinned memory metadata.
        Pointer p;
        auto res = cudaHostAlloc(reinterpret_cast<void **>(&p), numBytes + SD_ALLOC_PADDING, cudaHostAllocDefault);
        if (res != 0) throw cuda_exception::build("Can't allocate [HOST] memory", res);

        _mutexSpills.lock();
        _spillsSecondary.push_back(p);
        _mutexSpills.unlock();

        _spillsSizeSecondary += numBytes;

        return p;
      }

      result = (void *)(_ptrHost + _offsetSecondary.load());
      _offsetSecondary += numBytes;
      // memset(result, 0, (int) numBytes);

      sd_debug("Allocating %lld bytes from [HOST] workspace; Current PTR: %p; Current offset: %lld\n", numBytes, result,
               _offset.load());

      this->_mutexAllocation.unlock();

      return result;
    } break;
    case DEVICE: {
      if (numBytes < 1)
        throw allocation_exception::build("Number of [DEVICE] bytes for allocation should be positive", numBytes);

      // numBytes += 32;
      void *result = nullptr;
      this->_cycleAllocations += numBytes;
      this->_mutexAllocation.lock();

      if (_offset.load() + numBytes > _currentSize) {
        sd_debug("Allocating %lld [DEVICE] bytes in spills\n", numBytes);
        this->_mutexAllocation.unlock();

        int deviceId = 0;
        cudaGetDevice(&deviceId);
        Pointer p = CudaMemoryPool::getInstance().allocate(numBytes, deviceId, currentStream());
        if (p == nullptr) {
          // GPU OOM: fall back to pinned host memory (accessible from GPU via unified addressing)
          sd_debug("DEVICE OOM - falling back to pinned host workspace for %lld bytes\n", numBytes);
          // Re-route to HOST allocation path to keep accounting consistent
          return allocateBytes(HOST, numBytes);
        }

        _mutexSpills.lock();
        _spills.push_back(p);
        _mutexSpills.unlock();

        _spillsSize += numBytes;

        return p;
      }

      result = (void *)(_ptrDevice + _offset.load());
      _offset += numBytes;
      // memset(result, 0, (int) numBytes);

      sd_debug("Allocating %lld bytes from [DEVICE] workspace; Current PTR: %p; Current offset: %lld\n", numBytes,
               result, _offset.load());

      this->_mutexAllocation.unlock();

      return result;
    } break;
    default:
      THROW_EXCEPTION("Unknown MemoryType was passed in");
  }
}

Workspace *Workspace::clone() {
  // for clone we take whatever is higher: current allocated size, or allocated size of current loop
  return new Workspace(sd::math::sd_max<LongType>(this->getCurrentSize(), this->_cycleAllocations.load()));
}

LongType Workspace::getAllocatedSecondarySize() { return getCurrentSecondarySize() + getSpilledSecondarySize(); }

LongType Workspace::getCurrentSecondarySize() { return _currentSizeSecondary; }

LongType Workspace::getCurrentSecondaryOffset() { return _offsetSecondary.load(); }

LongType Workspace::getSpilledSecondarySize() { return _spillsSizeSecondary; }

LongType Workspace::getUsedSecondarySize() { return getCurrentSecondaryOffset(); }

}  // namespace memory
}  // namespace sd
