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
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <execution/ContextBuffers.h>
#include <string>
#include <helpers/logger.h>
#include <memory/cuda/CudaMemoryPool.h>

namespace sd {

// Helper: get the current CUDA device from the runtime (not thread affinity).
// This respects whatever device DeviceMemoryManager.switchDevice() set.
static int currentCudaDevice() {
  int dev = 0;
  cudaGetDevice(&dev);
  return dev;
}

ContextBuffers::ContextBuffers() {
  _deviceId = currentCudaDevice();
}

ContextBuffers::ContextBuffers(const ContextBuffers& other) {
  release();

  this->_initialized = other._initialized;
  // Do NOT copy _allocated - only one object should own the resources.
  // The original object keeps ownership, this copy is just a view.
  this->_allocated = false;
  this->_deviceId = other._deviceId;

  this->_specialStream = other._specialStream;
  this->_execStream = other._execStream;
  this->_allocationPointer = other._allocationPointer;
  this->_reductionPointer = other._reductionPointer;
  this->_scalarPointer = other._scalarPointer;
}

ContextBuffers& ContextBuffers::operator=(const ContextBuffers& other) {
  release();

  this->_initialized = other._initialized;
  // Do NOT copy _allocated - only one object should own the resources.
  // The original object keeps ownership, this copy is just a view.
  this->_allocated = false;
  this->_deviceId = other._deviceId;

  this->_specialStream = other._specialStream;
  this->_execStream = other._execStream;
  this->_allocationPointer = other._allocationPointer;
  this->_reductionPointer = other._reductionPointer;
  this->_scalarPointer = other._scalarPointer;

  return *this;
}

ContextBuffers& ContextBuffers::operator=(ContextBuffers&& other) {
  release();

  this->_initialized = other._initialized;
  this->_allocated = other._allocated;
  this->_deviceId = other._deviceId;

  this->_specialStream = other._specialStream;
  this->_execStream = other._execStream;
  this->_allocationPointer = other._allocationPointer;
  this->_reductionPointer = other._reductionPointer;
  this->_scalarPointer = other._scalarPointer;

  // Transfer ownership - the moved-from object no longer owns resources
  other._allocated = false;
  other._initialized = false;
  other._specialStream = nullptr;
  other._execStream = nullptr;
  other._allocationPointer = nullptr;
  other._reductionPointer = nullptr;
  other._scalarPointer = nullptr;

  return *this;
}

void ContextBuffers::release() {
  if (_allocated) {
    // Must be on the correct device to sync/destroy streams and free memory.
    // Streams and device memory are only valid on the device they were created on.
    int currentDevice = -1;
    auto getDevRes = cudaGetDevice(&currentDevice);

    // If CUDA is shutting down, just clean up C++ memory
    if (getDevRes != cudaSuccess) {
      if (_execStream != nullptr) delete reinterpret_cast<cudaStream_t*>(_execStream);
      if (_specialStream != nullptr) delete reinterpret_cast<cudaStream_t*>(_specialStream);
      _allocated = false;
      _deviceId = -1;
      this->_specialStream = nullptr;
      this->_execStream = nullptr;
      this->_allocationPointer = nullptr;
      this->_reductionPointer = nullptr;
      this->_scalarPointer = nullptr;
      _initialized = false;
      return;
    }

    // Switch to the context's device if needed
    bool switchedDevice = false;
    if (_deviceId >= 0 && currentDevice != _deviceId) {
      cudaSetDevice(_deviceId);
      switchedDevice = true;
    }

    // Free workspace buffers — routed through the pool so that both
    // cudaMallocAsync (primary path) and allocateDirect (906 fallback)
    // allocations are freed correctly regardless of which path was taken.
    if (_allocationPointer != nullptr) {
      memory::CudaMemoryPool::getInstance().free(_allocationPointer, _deviceId);
    }
    if (_scalarPointer != nullptr) cudaFreeHost(_scalarPointer);
    if (_reductionPointer != nullptr) {
      memory::CudaMemoryPool::getInstance().free(_reductionPointer, _deviceId);
    }

    if (_execStream != nullptr) {
      auto _cudaStream = reinterpret_cast<cudaStream_t*>(_execStream);
      if (*_cudaStream != nullptr) {
        memory::CudaMemoryPool::getInstance().removeDirtyStream(_deviceId, *_cudaStream);
        // Do NOT sync or destroy streams here. Pending async operations on these streams
        // may reference memory that is about to be freed or has already been freed (e.g.,
        // DSP graph replay, cudaFreeAsync). Synchronizing would execute those operations
        // and trigger illegal memory access (error 700). Destroying corrupts CUDA context
        // causing "invalid resource handle" (error 900) on subsequent operations.
        // Stream handles are leaked; resources reclaimed at process exit.
      }
      delete _cudaStream;
    }

    if (_specialStream != nullptr) {
      auto _cudaSpecialStream = reinterpret_cast<cudaStream_t*>(_specialStream);
      if (*_cudaSpecialStream != nullptr) {
        memory::CudaMemoryPool::getInstance().removeDirtyStream(_deviceId, *_cudaSpecialStream);
      }
      delete _cudaSpecialStream;
    }

    // Restore original device if we switched
    if (switchedDevice && currentDevice >= 0) {
      cudaSetDevice(currentDevice);
    }

    // Clear any errors that may have occurred during release.
    cudaGetLastError();

    _allocated = false;
    _deviceId = -1;
    this->_specialStream = nullptr;
    this->_execStream = nullptr;
    this->_allocationPointer = nullptr;
    this->_reductionPointer = nullptr;
    this->_scalarPointer = nullptr;
  }

  _initialized = false;
}

ContextBuffers::~ContextBuffers() { release(); }

ContextBuffers::ContextBuffers(void* rPointer, void* sPointer, void* aPointer, bool isOwner) {
  _reductionPointer = rPointer;
  _scalarPointer = sPointer;
  _allocationPointer = aPointer;
  _allocated = isOwner;
}

void ContextBuffers::initialize() {
  // Use the current CUDA device (set by DeviceMemoryManager.switchDevice on the Java side).
  // Do NOT use AffinityManager — device selection is handled by DeviceMemoryManager.
  _deviceId = currentCudaDevice();
  cudaSetDevice(_deviceId);

  // Clear any previous CUDA errors before attempting allocations.
  cudaError_t prevErr = cudaGetLastError();
  if (prevErr != cudaSuccess) {
    sd_debug("ContextBuffers::initialize: Cleared previous CUDA error: %s (device %d)\n",
             cudaGetErrorString(prevErr), _deviceId);
  }

  // Trim pool to reclaim freed-but-reserved memory on this device.
  memory::CudaMemoryPool::getInstance().trimPool(_deviceId);
  cudaGetLastError();  // clear any error from trim

  // Allocate workspace buffers using cudaMallocAsync on the default pool.
  //  We must NOT use CudaMemoryPool::allocate() here because its
  // allocateFailover() silently routes to a different device when the current
  // device is low on memory. This creates a fatal mismatch: ContextBuffers
  // workspace and streams end up on device 1, but ops use device 0 data,
  // causing "illegal memory access" (error 700) and "invalid resource handle"
  // (error 900) crashes. ContextBuffers MUST stay on the requested device.
  //
  // We use cudaMallocAsync instead of cudaMalloc because the CUDA memory pool
  // may have reserved most of the device memory. cudaMalloc allocates from
  // non-pool memory and will fail if the pool has reserved everything.
  // cudaMallocAsync allocates FROM the pool and can use pool-reserved memory.
  // We use stream 0 (default stream) to ensure the allocation is immediately
  // available on any stream.
  //
  // If cudaMallocAsync fails with error 906 (cudaErrorStreamCaptureImplicit),
  // we're being called during CUDA graph capture. Fall back to cudaMalloc
  // which doesn't participate in stream ordering and won't invalidate capture.
  auto res = cudaMallocAsync(&_reductionPointer, 1024 * 1024 * 8, 0);
  if (res == 906) {
    // Error 906 = cudaErrorStreamCaptureImplicit: stream 0 would depend on
    // a capturing blocking stream. Use allocateDirect instead — it allocates
    // on a dedicated non-capturing stream so it is safe during graph capture
    // and produces a persistent buffer that survives capture/teardown.
    cudaGetLastError();  // clear error 906
    _reductionPointer = memory::CudaMemoryPool::getInstance().allocateDirect(1024 * 1024 * 8, _deviceId);
    res = (_reductionPointer != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
  }
  if (res != cudaSuccess) {
    _reductionPointer = nullptr;
    // OOM on this device — log warning but do NOT throw.
    // Throwing from ContextBuffers initialization (thread-local storage) causes
    // terminate() → SIGABRT because the exception propagates through noexcept
    // boundaries in thread destructors or TLS initialization.
    // Ops that need reduction buffers will check for null and fail gracefully.
    sd_printf("WARNING: ContextBuffers: _reductionPointer allocation failed on device %d "
              "(error %d: %s). Ops requiring reduction/allocation buffers will fail on this device.\n",
              _deviceId, (int)res, cudaGetErrorString(res));
    cudaGetLastError();  // clear error
    _scalarPointer = nullptr;
    _allocationPointer = nullptr;
    // CRITICAL: always create streams even when buffer allocation fails.
    // getCudaStream() → execStream() → _execStream. If _execStream stays null here,
    // LaunchContext::getCudaStream() returns null, and DataBuffer::syncToPrimary
    // null-derefs *streamPtr → SIGSEGV si_addr=0x0.
    // Streams do NOT participate in pool memory or CUDA graph capture — cudaStreamCreate
    // is always safe regardless of pool/capture state.
    _execStream = new cudaStream_t();
    _specialStream = new cudaStream_t();
    cudaGetLastError();  // clear any sticky error before stream creation
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_execStream));
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_specialStream));
    // Set _allocated=true so release()/dtor runs the stream-cleanup block and deletes
    // the cudaStream_t objects. The buffer pointers are all null so free/cudaFreeHost
    // on them are no-ops (both check != nullptr before freeing).
    _allocated = true;
    _initialized = true;
    return;
  }

  res = cudaHostAlloc(reinterpret_cast<void**>(&_scalarPointer), 16, cudaHostAllocDefault);
  if (res != cudaSuccess) {
    memory::CudaMemoryPool::getInstance().free(_reductionPointer, _deviceId);
    _reductionPointer = nullptr;
    _scalarPointer = nullptr;
    _allocationPointer = nullptr;
    sd_printf("WARNING: ContextBuffers: _scalarPointer allocation failed on device %d "
              "(error %d: %s)\n", _deviceId, (int)res, cudaGetErrorString(res));
    cudaGetLastError();
    // CRITICAL: same as the _reductionPointer path above — create streams so that
    // getCudaStream() never returns null and syncToPrimary does not null-deref.
    // Set _allocated=true so release()/dtor cleans up the stream objects.
    _execStream = new cudaStream_t();
    _specialStream = new cudaStream_t();
    cudaGetLastError();
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_execStream));
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_specialStream));
    _allocated = true;
    _initialized = true;
    return;
  }

  res = cudaMallocAsync(&_allocationPointer, 1024 * 1024 * 8, 0);
  if (res == 906) {
    // Error 906 = cudaErrorStreamCaptureImplicit: same capture-safe fallback
    // as for _reductionPointer above — allocateDirect uses a dedicated
    // non-capturing stream, producing a persistent buffer.
    cudaGetLastError();
    _allocationPointer = memory::CudaMemoryPool::getInstance().allocateDirect(1024 * 1024 * 8, _deviceId);
    res = (_allocationPointer != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
  }
  if (res != cudaSuccess) {
    memory::CudaMemoryPool::getInstance().free(_reductionPointer, _deviceId);
    _reductionPointer = nullptr;
    cudaFreeHost(_scalarPointer);
    _scalarPointer = nullptr;
    _allocationPointer = nullptr;
    sd_printf("WARNING: ContextBuffers: _allocationPointer allocation failed on device %d "
              "(error %d: %s)\n", _deviceId, (int)res, cudaGetErrorString(res));
    cudaGetLastError();
    // CRITICAL: same as the other early-return paths — create streams so that
    // getCudaStream() never returns null and syncToPrimary does not null-deref.
    // Set _allocated=true so release()/dtor cleans up the stream objects.
    _execStream = new cudaStream_t();
    _specialStream = new cudaStream_t();
    cudaGetLastError();
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_execStream));
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_specialStream));
    _allocated = true;
    _initialized = true;
    return;
  }
  // Sync default stream to ensure async allocations are complete before use.
  cudaStreamSynchronize(0);

  _execStream = new cudaStream_t();
  _specialStream = new cudaStream_t();
  if (nullptr == _execStream || nullptr == _specialStream)
    THROW_EXCEPTION("Failed to allocate memory for new CUDA stream");

  // Clear any sticky CUDA error before stream creation.
  cudaGetLastError();

  res = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_execStream));
  if (res != cudaSuccess) {
    std::string msg = "Failed to create default CUDA stream with launch context; Error code: [" + std::to_string(res) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  res = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_specialStream));
  if (res != cudaSuccess) {
    std::string msg = "Failed to create special CUDA stream with launch context; Error code: [" + std::to_string(res) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  _allocated = true;
  _initialized = true;
}

void* ContextBuffers::reductionBuffer() {
  // No flip-flop: ContextBuffers stays on its device. Per-device routing is
  // handled by LaunchContext's per-device ContextBuffers map.
  if (!_initialized) {
    initialize();
  }

  return _reductionPointer;
}

void* ContextBuffers::scalarBuffer() {
  if (!_initialized) {
    initialize();
  }

  return _scalarPointer;
}

void* ContextBuffers::allocationBuffer() {
  if (!_initialized) {
    initialize();
  }

  return _allocationPointer;
}

void ContextBuffers::setReductionBuffer(void* pointer) { _reductionPointer = pointer; }

void ContextBuffers::setScalarBuffer(void* pointer) { _scalarPointer = pointer; }

void ContextBuffers::setAllocationBuffer(void* pointer) { _allocationPointer = pointer; }

void ContextBuffers::triggerOwnership(bool isOwner) { _allocated = isOwner; }

int ContextBuffers::deviceId() { return _deviceId; }

void* ContextBuffers::execStream() {
  if (!_initialized) {
    initialize();
  }

  return _execStream;
}

void* ContextBuffers::specialStream() {
  if (!_initialized) {
    initialize();
  }

  return _specialStream;
}

bool ContextBuffers::isInitialized() { return _initialized; }

ErrorReference* ContextBuffers::errorReference() { return &_errorReference; }
}  // namespace sd
