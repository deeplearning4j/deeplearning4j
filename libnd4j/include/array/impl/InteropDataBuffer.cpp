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
#include <array/DataTypeUtils.h>
#include <array/InteropDataBuffer.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/logger.h>
#include <helpers/TransferMetrics.h>

namespace sd {

// ============================================================================
// BufferAccessGuard implementation
// ============================================================================

BufferAccessGuard::BufferAccessGuard(const InteropDataBuffer* buffer)
    : _buffer(buffer), _primaryPtr(nullptr), _specialPtr(nullptr), _valid(false) {
  if (_buffer == nullptr) {
    return;
  }

  // Validate magic number before accessing
  if (_buffer->_magic != INTEROP_BUFFER_MAGIC) {
    _buffer = nullptr;
    return;
  }

  // Try to acquire access
  if (!_buffer->acquireAccess()) {
    _buffer = nullptr;
    return;
  }

  // Access acquired - cache the pointers while holding access
  _valid = true;
  _primaryPtr = _buffer->primaryNoRelease();
  _specialPtr = _buffer->specialNoRelease();
  // NOTE: We do NOT release access here - that's the whole point!
  // Access will be released in destructor or release()
}

BufferAccessGuard::~BufferAccessGuard() {
  release();
}

BufferAccessGuard::BufferAccessGuard(BufferAccessGuard&& other) noexcept
    : _buffer(other._buffer),
      _primaryPtr(other._primaryPtr),
      _specialPtr(other._specialPtr),
      _valid(other._valid) {
  // Invalidate source - we now own the access
  other._buffer = nullptr;
  other._primaryPtr = nullptr;
  other._specialPtr = nullptr;
  other._valid = false;
}

BufferAccessGuard& BufferAccessGuard::operator=(BufferAccessGuard&& other) noexcept {
  if (this != &other) {
    // Release our current access
    release();

    // Take ownership from other
    _buffer = other._buffer;
    _primaryPtr = other._primaryPtr;
    _specialPtr = other._specialPtr;
    _valid = other._valid;

    // Invalidate source
    other._buffer = nullptr;
    other._primaryPtr = nullptr;
    other._specialPtr = nullptr;
    other._valid = false;
  }
  return *this;
}

void BufferAccessGuard::release() {
  if (_valid && _buffer != nullptr) {
    _buffer->releaseAccess();
  }
  _buffer = nullptr;
  _primaryPtr = nullptr;
  _specialPtr = nullptr;
  _valid = false;
}

// ============================================================================
// InteropDataBuffer new methods
// ============================================================================

BufferAccessGuard InteropDataBuffer::acquireGuard() const {
  return BufferAccessGuard(this);
}

void* InteropDataBuffer::primaryNoRelease() const {
  // This method is ONLY called after acquireAccess() succeeded
  // It does NOT call releaseAccess() - the caller (BufferAccessGuard) handles that

  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if (db == nullptr) {
    return nullptr;
  }
  void* result = db->primary();
  return result != nullptr ? reinterpret_cast<int8_t*>(result) : nullptr;
}

void* InteropDataBuffer::specialNoRelease() const {
  // This method is ONLY called after acquireAccess() succeeded
  // It does NOT call releaseAccess() - the caller (BufferAccessGuard) handles that

  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if (db == nullptr) {
    return nullptr;
  }
  void* result = db->special();
  return result != nullptr ? reinterpret_cast<int8_t*>(result) : nullptr;
}

// ============================================================================
// Original InteropDataBuffer implementation
// ============================================================================
InteropDataBuffer::InteropDataBuffer(InteropDataBuffer* dataBuffer, uint64_t length) {
 if(dataBuffer == nullptr) {
        THROW_EXCEPTION("InteropDataBuffer::InteropDataBuffer(InteropDataBuffer& dataBuffer, uint64_t length, uint64_t offset) - dataBuffer is nullptr");
 }
  DataBuffer* srcBuffer = dataBuffer->dataBuffer();
  if(srcBuffer == nullptr || srcBuffer->getDataType() == DataType::UNKNOWN)
    THROW_EXCEPTION("InteropDataBuffer::InteropDataBuffer(InteropDataBuffer& dataBuffer, uint64_t length, uint64_t offset) - dataBuffer has unknown data type");
  _dataBuffer.store(srcBuffer, std::memory_order_release);
  _dataType = dataBuffer->_dataType;

  _cachedLenInBytes = length * DataTypeUtils::sizeOf(_dataType);
  // Cache pointers for deallocation tracking
  if (srcBuffer != nullptr) {
    _cachedPrimaryPtr = srcBuffer->primary();
    _cachedSpecialPtr = srcBuffer->special();
  }

  owner = false;
}

InteropDataBuffer::InteropDataBuffer(DataBuffer * databuffer) {
  _dataBuffer.store(databuffer, std::memory_order_release);
  _dataType = databuffer->getDataType();
  if(_dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION(
        "InteropDataBuffer::InteropDataBuffer(size_t lenInBytes, DataType dtype, bool allocateBoth) - data type is unknown");
  }
  // Cache the size to avoid accessing freed memory later
  _cachedLenInBytes = databuffer != nullptr ? databuffer->getLenInBytes() : 0;
  // Cache pointers for deallocation tracking
  if (databuffer != nullptr) {
    _cachedPrimaryPtr = databuffer->primary();
    _cachedSpecialPtr = databuffer->special();
  }
  // When wrapping an existing DataBuffer, we don't own it by default
  owner = false;
}

InteropDataBuffer::InteropDataBuffer(size_t lenInBytes, DataType dtype, bool allocateBoth) {
  if(dtype == DataType::UNKNOWN) {
    THROW_EXCEPTION(
        "InteropDataBuffer::InteropDataBuffer(size_t lenInBytes, DataType dtype, bool allocateBoth) - data type is unknown");
  }

  _cachedLenInBytes = lenInBytes;

  // FIX: Always create a DataBuffer, even for zero-length arrays.
  // External data buffers (via dbCreateExternalDataBuffer) need a DataBuffer
  // to manage external pointers set via setPrimary/setSpecial.
  // The DataBuffer is created with lenInBytes=0 to avoid DEVICE allocation,
  // and the actual length will be set by setPrimary/setSpecial.
  DataBuffer* newBuffer = new DataBuffer(lenInBytes, dtype, nullptr, allocateBoth);
  _dataBuffer.store(newBuffer, std::memory_order_release);
  this->_dataType = dtype;
  this->markOwner(true);
  // Cache pointers for deallocation tracking
  _cachedPrimaryPtr = newBuffer->primary();
  _cachedSpecialPtr = newBuffer->special();
}


void InteropDataBuffer::printDbAllocationTrace() {
  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if(db == nullptr)
    return;
  db->printAllocationTrace();
}

void InteropDataBuffer::markOwner(bool owner) {
  this->owner = owner;
  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if(db != nullptr && !isClosed()) {
    db->_isOwnerPrimary = owner;
    db->_isOwnerSpecial = owner;
  }
}

DataBuffer * InteropDataBuffer::getDataBuffer() const {
  // Validate magic number to detect use-after-free
  if (_magic != INTEROP_BUFFER_MAGIC) {
    std::string errorMessage = "InteropDataBuffer::getDataBuffer(): USE-AFTER-FREE DETECTED! ";
    if (_magic == INTEROP_BUFFER_FREED) {
      errorMessage += "Buffer was explicitly freed. ";
    } else {
      errorMessage += "Magic number corrupted - memory reused. ";
    }
    THROW_EXCEPTION(errorMessage.c_str());
  }

  //this can effect size of calculations among others
  if(_dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("All interop buffers must have a known data type.");
  }

  // Return nullptr if closed to prevent use-after-free
  // EXCEPTION: Constant buffers remain valid even after _closed is set,
  // because their data is never freed and their pointer is never nulled.
  bool bufferIsConstant = isConstant.load(std::memory_order_acquire);
  if (isClosed() && !bufferIsConstant) {
    return nullptr;
  }

  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);

  // Set data type if needed (this is a non-const operation on the DataBuffer,
  // which is safe because DataBuffer's _dataType is independent of the atomic pointer)
  if(db != nullptr && db->_dataType == DataType::UNKNOWN) {
    db->_dataType = _dataType;
  }

  return db;
}

DataBuffer * InteropDataBuffer::dataBuffer() {
  // Validate magic number to detect use-after-free
  if (_magic != INTEROP_BUFFER_MAGIC) {
    std::string errorMessage = "InteropDataBuffer::dataBuffer(): USE-AFTER-FREE DETECTED! ";
    if (_magic == INTEROP_BUFFER_FREED) {
      errorMessage += "Buffer was explicitly freed. ";
    } else {
      errorMessage += "Magic number corrupted - memory reused. ";
    }
    THROW_EXCEPTION(errorMessage.c_str());
  }

  bool bufferIsConstant = isConstant.load(std::memory_order_acquire);
  if(isClosed() && !bufferIsConstant) {
    return nullptr;
  }

  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  return db;
}



void* InteropDataBuffer::primary() const {
  if (_magic != INTEROP_BUFFER_MAGIC) {
    std::string errorMessage = "InteropDataBuffer::primary(): USE-AFTER-FREE DETECTED! ";
    if (_magic == INTEROP_BUFFER_FREED) {
      errorMessage += "Buffer was explicitly freed (destructor was called). ";
    } else {
      errorMessage += "Magic number corrupted (0x";
      char hexBuf[32];
      snprintf(hexBuf, sizeof(hexBuf), "%lx", static_cast<unsigned long>(_magic));
      errorMessage += hexBuf;
      errorMessage += ") - memory was freed and reused. ";
    }
    errorMessage += "This indicates a dangling pointer to a closed/freed InteropDataBuffer.";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!acquireAccess()) {
    return nullptr;
  }

  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);

  void* result = nullptr;
  if(db != nullptr) {
    result = db->primary();
  }

  releaseAccess();

  return result != nullptr ? reinterpret_cast<int8_t*>(result) : nullptr;
}

void* InteropDataBuffer::special() const {
  if (_magic != INTEROP_BUFFER_MAGIC) {
    std::string errorMessage = "InteropDataBuffer::special(): USE-AFTER-FREE DETECTED! ";
    if (_magic == INTEROP_BUFFER_FREED) {
      errorMessage += "Buffer was explicitly freed (destructor was called). ";
    } else {
      errorMessage += "Magic number corrupted (0x";
      char hexBuf[32];
      snprintf(hexBuf, sizeof(hexBuf), "%lx", static_cast<unsigned long>(_magic));
      errorMessage += hexBuf;
      errorMessage += ") - memory was freed and reused. ";
    }
    errorMessage += "This indicates a dangling pointer to a closed/freed InteropDataBuffer.";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!acquireAccess()) {
    return nullptr;
  }

  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);

  void* result = nullptr;
  if(db != nullptr) {
    result = db->special();
  }

  releaseAccess();

  return result != nullptr ? reinterpret_cast<int8_t*>(result) : nullptr;
}

void InteropDataBuffer::setSpecial(void* ptr, size_t length) {
  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if(db == nullptr)
    THROW_EXCEPTION("InteropDataBuffer::setSpecial() - _dataBuffer is nullptr");
  if(isClosed())
    return;  // Silently ignore if buffer was already closed
  db->setSpecialBuffer(ptr, length);
  // Update cached pointer
  _cachedSpecialPtr = ptr;
}

void InteropDataBuffer::setPrimary(void* ptr, size_t length) {
  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if(db == nullptr)
    THROW_EXCEPTION("InteropDataBuffer::setPrimary() - _dataBuffer is nullptr");
  if(isClosed())
    return;  // Silently ignore if buffer was already closed
  db->setPrimaryBuffer(ptr, length);
  // Update cached pointer
  _cachedPrimaryPtr = ptr;
}

void InteropDataBuffer::setDeviceId(int deviceId) {
  if(isClosed())
    return;
  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if(db == nullptr)
    return;
  db->setDeviceId(deviceId);
}

int InteropDataBuffer::deviceId() const {
  // Constant buffers remain valid even after _closed is set
  bool bufferIsConstant = isConstant.load(std::memory_order_acquire);
  if(isClosed() && !bufferIsConstant)
    return 0;
  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if(db == nullptr)
    return 0;
  return db->deviceId();
}

int InteropDataBuffer::useCount() const {
  return 1;
}

void InteropDataBuffer::registerSpecialUse(const std::vector<const InteropDataBuffer*>& writeList,
                                           const std::vector<const InteropDataBuffer*>& readList) {
  // Advance readSpecial on read inputs — matches NDArray::registerSpecialUse
  // (NDArray_core.cu) which calls tickReadDevice() on readList items.
  // Without this, isPrimaryActual() returns incorrect results because
  // _readSpecial is never advanced, causing stale D→H sync decisions.
  for (const auto& v : readList) {
    if (v == nullptr) continue;
    auto db = v->getDataBuffer();
    if (db != nullptr) db->readSpecial();
  }

  for (const auto& v : writeList) {
    if (v == nullptr) continue;

    v->getDataBuffer()->writeSpecial();

    // Update multi-backend coherence: device (special) was written,
    // so mark CUDA device as MODIFIED and invalidate CPU
    if (v->_multiBackendWorkspace != nullptr) {
      auto currentDeviceId = AffinityManager::currentDeviceId();
      memory::DeviceDescriptor gpuDevice(memory::DeviceType::CUDA_GPU, currentDeviceId);
      v->_multiBackendWorkspace->markModified(gpuDevice);
    }
  }
}

void InteropDataBuffer::prepareSpecialUse(const std::vector<const InteropDataBuffer*>& writeList,
                                          const std::vector<const InteropDataBuffer*>& readList,
                                          bool synchronizeWritables) {
  auto currentDeviceId = AffinityManager::currentDeviceId();
  memory::DeviceDescriptor gpuDevice(memory::DeviceType::CUDA_GPU, currentDeviceId);

  for (const auto& v : readList) {
    if (v == nullptr || v->isClosed()) continue;

    auto db = v->getDataBuffer();
    if(db == nullptr) continue;

    if (db->deviceId() != currentDeviceId) db->migrate();

    // If multi-backend workspace coherence says GPU already has valid data,
    // skip the potentially expensive H2D sync
    if (v->_multiBackendWorkspace != nullptr) {
      auto coherence = v->_multiBackendWorkspace->getCoherenceState(gpuDevice);
      if (coherence == memory::CoherenceState::SHARED ||
          coherence == memory::CoherenceState::EXCLUSIVE ||
          coherence == memory::CoherenceState::MODIFIED) {
        continue;  // GPU data is already valid
      }
    }

    db->syncToSpecial();
  }

  // we don't tick write list, only ensure the same device affinity
  for (const auto& v : writeList) {
    if (v == nullptr || v->isClosed()) continue;

    auto db = v->getDataBuffer();
    if(db == nullptr) continue;

    // special case for legacy ops - views can be updated on host side, thus original array can be not updated
    if (!db->isSpecialActual()) {
      // Check coherence before syncing
      if (v->_multiBackendWorkspace != nullptr) {
        auto coherence = v->_multiBackendWorkspace->getCoherenceState(gpuDevice);
        if (coherence != memory::CoherenceState::SHARED &&
            coherence != memory::CoherenceState::EXCLUSIVE &&
            coherence != memory::CoherenceState::MODIFIED) {
          db->syncToSpecial();
        }
      } else {
        db->syncToSpecial();
      }
    }

    if (db->deviceId() != currentDeviceId) db->migrate();
  }
}

void InteropDataBuffer::registerPrimaryUse(const std::vector<const InteropDataBuffer*>& writeList,
                                           const std::vector<const InteropDataBuffer*>& readList) {
  // Advance readPrimary on read inputs — matches NDArray::registerPrimaryUse
  // (NDArray_core.cu) which calls tickReadHost() on readList items.
  for (const auto& v : readList) {
    if (v == nullptr || v->isClosed()) continue;
    auto db = v->getDataBuffer();
    if (db != nullptr) db->readPrimary();
  }

  for (const auto& v : writeList) {
    if (v == nullptr || v->isClosed()) continue;

    auto db = v->getDataBuffer();
    if (db != nullptr) {
      db->writePrimary();

      // Update multi-backend coherence: host (primary) was written,
      // so mark CPU as MODIFIED and invalidate GPU devices
      if (v->_multiBackendWorkspace != nullptr) {
        memory::DeviceDescriptor cpuDevice(memory::DeviceType::CPU, 0);
        v->_multiBackendWorkspace->markModified(cpuDevice);
      }
    }
  }
}

void InteropDataBuffer::preparePrimaryUse(const std::vector<const InteropDataBuffer*>& writeList,
                                          const std::vector<const InteropDataBuffer*>& readList,
                                          bool synchronizeWritables) {
  memory::DeviceDescriptor cpuDevice(memory::DeviceType::CPU, 0);

  // Sync read buffers from device to host
  for (const auto& v : readList) {
    if (v == nullptr || v->isClosed()) continue;

    auto db = v->getDataBuffer();
    if (db == nullptr) continue;

    // If multi-backend workspace coherence says CPU already has valid data,
    // skip the potentially expensive D2H sync
    if (v->_multiBackendWorkspace != nullptr) {
      auto coherence = v->_multiBackendWorkspace->getCoherenceState(cpuDevice);
      if (coherence == memory::CoherenceState::SHARED ||
          coherence == memory::CoherenceState::EXCLUSIVE ||
          coherence == memory::CoherenceState::MODIFIED) {
        continue;  // CPU data is already valid
      }
    }

    // Transfer metrics are tracked inside syncToPrimary
    db->syncToPrimary(LaunchContext::defaultContext(), false);
  }

  // Ensure write buffers have primary allocation
  for (const auto& v : writeList) {
    if (v == nullptr || v->isClosed()) continue;

    auto db = v->getDataBuffer();
    if (db == nullptr) continue;

    db->allocatePrimary();
    if (synchronizeWritables && !db->isPrimaryActual()) {
      // Check coherence before syncing
      if (v->_multiBackendWorkspace != nullptr) {
        auto coherence = v->_multiBackendWorkspace->getCoherenceState(cpuDevice);
        if (coherence != memory::CoherenceState::SHARED &&
            coherence != memory::CoherenceState::EXCLUSIVE &&
            coherence != memory::CoherenceState::MODIFIED) {
          db->syncToPrimary(LaunchContext::defaultContext(), false);
        }
      } else {
        db->syncToPrimary(LaunchContext::defaultContext(), false);
      }
    }
  }
}

void InteropDataBuffer::setWorkspace(memory::Workspace* workspace) {
  _workspace = workspace;
  if (workspace != nullptr) {
    // Workspace owns the memory, so this buffer should not free it
    markOwner(false);
  }
}

void InteropDataBuffer::expand(size_t newlength) {
  if(isClosed())
    return;  // Cannot expand a closed buffer
  DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);
  if(db == nullptr)
    return;  // Cannot expand a null buffer
  db->expand(newlength * DataTypeUtils::sizeOf(db->getDataType()));
}


}  // namespace sd
