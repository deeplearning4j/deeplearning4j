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
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <execution/ContextBuffers.h>
#include <helpers/logger.h>
#include <memory/cuda/CudaMemoryPool.h>

namespace sd {
ContextBuffers::ContextBuffers() {
  _deviceId = AffinityManager::currentDeviceId();
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

    if (_allocationPointer != nullptr) {
      memory::CudaMemoryPool::getInstance().free(_allocationPointer, _deviceId, nullptr);
    }
    if (_scalarPointer != nullptr) cudaFreeHost(_scalarPointer);
    if (_reductionPointer != nullptr) {
      memory::CudaMemoryPool::getInstance().free(_reductionPointer, _deviceId, nullptr);
    }

    if (_execStream != nullptr) {
      auto _cudaStream = reinterpret_cast<cudaStream_t*>(_execStream);
      if (*_cudaStream != nullptr) {
        cudaStreamSynchronize(*_cudaStream);
        cudaStreamDestroy(*_cudaStream);
      }
      delete _cudaStream;
    }

    if (_specialStream != nullptr) {
      auto _cudaSpecialStream = reinterpret_cast<cudaStream_t*>(_specialStream);
      if (*_cudaSpecialStream != nullptr) {
        cudaStreamSynchronize(*_cudaSpecialStream);
        cudaStreamDestroy(*_cudaSpecialStream);
      }
      delete _cudaSpecialStream;
    }

    // Restore original device if we switched
    if (switchedDevice && currentDevice >= 0) {
      cudaSetDevice(currentDevice);
    }

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
  _deviceId = AffinityManager::currentNativeDeviceId();

  // Ensure we're on the correct device before allocating.
  // Without this, if reductionBuffer()/scalarBuffer()/allocationBuffer()
  // is called first (instead of execStream()), we might allocate on wrong device.
  cudaSetDevice(_deviceId);

  _reductionPointer = memory::CudaMemoryPool::getInstance().allocate(1024 * 1024 * 8, _deviceId, nullptr);
  if (_reductionPointer == nullptr) throw cuda_exception::build("_reductionPointer allocation failed", cudaErrorMemoryAllocation);

  auto res = cudaHostAlloc(reinterpret_cast<void**>(&_scalarPointer), 16, cudaHostAllocDefault);
  if (res != 0) throw cuda_exception::build("_scalarPointer allocation failed", res);

  _allocationPointer = memory::CudaMemoryPool::getInstance().allocate(1024 * 1024 * 8, _deviceId, nullptr);
  if (_allocationPointer == nullptr) throw cuda_exception::build("_allocationPointer allocation failed", cudaErrorMemoryAllocation);

  _execStream = new cudaStream_t();
  _specialStream = new cudaStream_t();
  if (nullptr == _execStream || nullptr == _specialStream)
    THROW_EXCEPTION("Failed to allocate memory for new CUDA stream");

  res = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_execStream));
  if (res != 0) throw cuda_exception::build("Failed to create default CUDA stream with launch context", res);

  res = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_specialStream));
  if (res != 0) throw cuda_exception::build("Failed to create special CUDA stream with launch context", res);

  _allocated = true;
  _initialized = true;
}

void* ContextBuffers::reductionBuffer() {
  int currentDevice = AffinityManager::currentNativeDeviceId();

  // Check if device changed since initialization - buffers are device-specific
  if (_initialized && _deviceId >= 0 && _deviceId != currentDevice) {
    release();
  }

  if (!_initialized) {
    cudaSetDevice(currentDevice);
    initialize();
  }

  return _reductionPointer;
}

void* ContextBuffers::scalarBuffer() {
  int currentDevice = AffinityManager::currentNativeDeviceId();

  // Check if device changed since initialization - buffers are device-specific
  if (_initialized && _deviceId >= 0 && _deviceId != currentDevice) {
    release();
  }

  if (!_initialized) {
    cudaSetDevice(currentDevice);
    initialize();
  }

  return _scalarPointer;
}

void* ContextBuffers::allocationBuffer() {
  int currentDevice = AffinityManager::currentNativeDeviceId();

  // Check if device changed since initialization - buffers are device-specific
  if (_initialized && _deviceId >= 0 && _deviceId != currentDevice) {
    release();
  }

  if (!_initialized) {
    cudaSetDevice(currentDevice);
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
  int currentDevice = AffinityManager::currentNativeDeviceId();

  // Check if device changed since initialization - streams are device-specific
  if (_initialized && _deviceId >= 0 && _deviceId != currentDevice) {
    // Release old streams (release() handles switching to correct device)
    release();
  }

  if (!_initialized) {
    // Make sure we're on the right device before initializing
    cudaSetDevice(currentDevice);
    initialize();
  }

  return _execStream;
}

void* ContextBuffers::specialStream() {
  int currentDevice = AffinityManager::currentNativeDeviceId();

  // Check if device changed since initialization - streams are device-specific
  if (_initialized && _deviceId >= 0 && _deviceId != currentDevice) {
    // Release old streams (release() handles switching to correct device)
    release();
  }

  if (!_initialized) {
    // Make sure we're on the right device before initializing
    cudaSetDevice(currentDevice);
    initialize();
  }

  return _specialStream;
}

bool ContextBuffers::isInitialized() { return _initialized; }

ErrorReference* ContextBuffers::errorReference() { return &_errorReference; }
}  // namespace sd
