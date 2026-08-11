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

/**
 * DataBuffer — Vulkan (SD_VULKAN) chip implementation.
 *
 * Implements the dual-buffer state machine (primary = host, special = device)
 * over VulkanMemoryPool. Every special allocation owns a real DEVICE_LOCAL
 * VkBuffer; unified-memory devices may expose that same memory type to the host,
 * but allocation placement never changes tier.
 *
 * The actuality counter semantics mirror array/cuda/DataBuffer.cu. H2D, D2H,
 * fill, and release update those counters only after the corresponding Vulkan
 * operation has completed successfully. DEVICE_LOCAL registry tokens are never
 * dereferenced on the host.
 *
 * @author (Vulkan port)
 */

// This translation unit is only meaningful for SD_VULKAN chip builds.
// When cmake's REMOVE_ITEM fails to exclude it (e.g. stale Makefile from a
// prior configure), the SD_VULKAN guard below ensures it compiles to an empty
// TU on CPU/CUDA builds rather than producing duplicate symbols or missing
// template instantiations.
#if defined(SD_VULKAN)

#if !defined(HAVE_VULKAN)
#error "SD_VULKAN requires a Vulkan SDK and HAVE_VULKAN=1"
#endif

#include <array/DataBuffer.h>
#include <array/DataTypeUtils.h>
#include <graph/DspDiagnostics.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <system/CanaryConstants.h>
#include <system/type_boilerplate.h>
#include <types/types.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif

// Vulkan handles remain encapsulated by the pool; DataBuffer never dereferences
// a DEVICE_LOCAL registry token on the host.

namespace sd {

// ─── Thread-local state (required by DataBuffer.h macros) ──────────────────────
SD_LIB_EXPORT DataBufferThreadState& dataBufferThreadState() {
  static thread_local DataBufferThreadState state;
  return state;
}

// ─── Internal helpers ───────────────────────────────────────────────────────────

namespace {

inline int currentVulkanDeviceId() {
  return sd::graph::VulkanDeviceManager::currentDeviceId();
}

inline int resolveVulkanAllocationDevice(
    void* pointer, std::atomic<int>& specialDeviceId,
    std::atomic<int>& deviceId, const char* invalidIdentityMessage) {
  if (pointer == nullptr) return -1;
  const int actualDevice =
      sd::graph::VulkanMemoryPool::getInstance().getDeviceId(pointer);
  if (actualDevice < 0) {
    THROW_EXCEPTION(invalidIdentityMessage);
  }

  int expectedDevice = -1;
  if (!specialDeviceId.compare_exchange_strong(
          expectedDevice, actualDevice, std::memory_order_acq_rel,
          std::memory_order_acquire) &&
      expectedDevice != actualDevice) {
    THROW_EXCEPTION(invalidIdentityMessage);
  }
  deviceId.store(actualDevice, std::memory_order_release);
  return actualDevice;
}

inline int validateVulkanAllocationDevice(
    void* pointer, const std::atomic<int>& specialDeviceId,
    const char* invalidIdentityMessage) {
  if (pointer == nullptr) return -1;
  const int actualDevice =
      sd::graph::VulkanMemoryPool::getInstance().getDeviceId(pointer);
  const int trackedDevice =
      specialDeviceId.load(std::memory_order_acquire);
  if (actualDevice < 0 ||
      (trackedDevice >= 0 && trackedDevice != actualDevice)) {
    THROW_EXCEPTION(invalidIdentityMessage);
  }
  return actualDevice;
}

inline VkDeviceSize checkedElementByteOffset(
    LongType elementOffset, LongType lengthBytes, size_t elementSize,
    const char* message) {
  if (elementOffset < 0 || lengthBytes < 0 || elementSize == 0) {
    THROW_EXCEPTION(message);
  }
  const auto offset = static_cast<VkDeviceSize>(elementOffset);
  const auto length = static_cast<VkDeviceSize>(lengthBytes);
  if (offset > length / static_cast<VkDeviceSize>(elementSize)) {
    THROW_EXCEPTION(message);
  }
  return offset * static_cast<VkDeviceSize>(elementSize);
}

inline VkDeviceSize checkedHostByteOffset(
    LongType elementOffset, size_t elementSize, const char* message) {
  if (elementOffset < 0 || elementSize == 0) {
    THROW_EXCEPTION(message);
  }
  const auto offset = static_cast<VkDeviceSize>(elementOffset);
  const auto size = static_cast<VkDeviceSize>(elementSize);
  if (offset > std::numeric_limits<VkDeviceSize>::max() / size) {
    THROW_EXCEPTION(message);
  }
  return offset * size;
}

}  // namespace

// ─── Actuality tick implementations ────────────────────────────────────────────
//
// Match CUDA's proven timestamp model exactly: every read or write takes
// the next global counter value, and actuality compares the latest timestamp
// observed on each side.

void DataBuffer::setCountersToZero() {
  _counter.store(0L);
  _writePrimary.store(0L);
  _writeSpecial.store(0L);
  _readPrimary.store(0L);
  _readSpecial.store(0L);
  _writeEventRecorded.store(false);
  _writeEventDeviceId.store(-1);
}

void DataBuffer::copyCounters(const DataBuffer& other) {
  _counter.store(other._counter);
  _writePrimary.store(other._readSpecial);
  _writeSpecial.store(other._readPrimary);
  _readPrimary.store(other._writeSpecial);
  _readSpecial.store(other._writePrimary);
}

void DataBuffer::writePrimary() const { _writePrimary = ++_counter; }

void DataBuffer::writeSpecial() const { _writeSpecial = ++_counter; }

void DataBuffer::readPrimary() const { _readPrimary = ++_counter; }

void DataBuffer::readSpecial() const { _readSpecial = ++_counter; }

bool DataBuffer::isPrimaryActual() const {
  return (_writePrimary.load(std::memory_order_acquire) > _writeSpecial.load(std::memory_order_acquire) ||
          _readPrimary.load(std::memory_order_acquire) > _writeSpecial.load(std::memory_order_acquire));
}

bool DataBuffer::isSpecialActual() const {
  return (_writeSpecial.load(std::memory_order_acquire) > _writePrimary.load(std::memory_order_acquire) ||
          _readSpecial.load(std::memory_order_acquire) > _writePrimary.load(std::memory_order_acquire));
}

// ─── Per-buffer stream ordering ───────────────────────────────────────────────

void DataBuffer::waitForSpecialWriteEvent(void* stream) const {
  if (!_writeEventRecorded.load(std::memory_order_acquire)) return;
  auto* event = sd::graph::VulkanExecutionEvent::fromOpaque(_writeEvent);
  if (event == nullptr) {
    THROW_EXCEPTION("DataBuffer::waitForSpecialWriteEvent: invalid Vulkan event");
  }
  const int deviceId = _writeEventDeviceId.load(std::memory_order_acquire);
  auto* target = stream == nullptr
      ? sd::graph::VulkanExecutionStream::defaultExecution(deviceId)
      : sd::graph::VulkanExecutionStream::fromOpaque(stream, false);
  if (target == nullptr || target->deviceId() != deviceId ||
      !target->waitEvent(*event)) {
    THROW_EXCEPTION("DataBuffer::waitForSpecialWriteEvent: Vulkan stream/event ordering failed");
  }
}

void DataBuffer::recordSpecialWriteEvent(void* stream) const {
  int deviceId = _specialDeviceId.load(std::memory_order_acquire);
  if (deviceId < 0) {
    THROW_EXCEPTION("DataBuffer::recordSpecialWriteEvent: missing Vulkan device identity");
  }
  auto* source = stream == nullptr
      ? sd::graph::VulkanExecutionStream::defaultExecution(deviceId)
      : sd::graph::VulkanExecutionStream::fromOpaque(stream, false);
  if (source == nullptr || source->deviceId() != deviceId) {
    THROW_EXCEPTION("DataBuffer::recordSpecialWriteEvent: invalid Vulkan stream");
  }
  auto* event = sd::graph::VulkanExecutionEvent::fromOpaque(_writeEvent);
  if (event == nullptr) {
    event = sd::graph::VulkanExecutionEvent::create();
    _writeEvent = event;
  }
  if (event == nullptr || !event->record(source)) {
    THROW_EXCEPTION("DataBuffer::recordSpecialWriteEvent: Vulkan event record failed");
  }
  _writeEventDeviceId.store(deviceId, std::memory_order_release);
  _writeEventRecorded.store(true, std::memory_order_release);
}

void DataBuffer::clearSpecialWriteEvent() const {
  auto* event = sd::graph::VulkanExecutionEvent::fromOpaque(_writeEvent);
  if (event != nullptr && !sd::graph::VulkanExecutionEvent::destroy(event)) {
    THROW_EXCEPTION("DataBuffer::clearSpecialWriteEvent: Vulkan event destroy failed");
  }
  _writeEvent = nullptr;
  _writeEventRecorded.store(false, std::memory_order_release);
  _writeEventDeviceId.store(-1, std::memory_order_release);
}

// ─── replaceSpecialBuffer ───────────────────────────────────────────────────────

void DataBuffer::replaceSpecialBuffer(void* newPtr, bool isOwner) {
  throwIfFrozen("replaceSpecialBuffer");
  int devId = -1;
  if (newPtr != nullptr) {
    devId = sd::graph::VulkanMemoryPool::getInstance().getDeviceId(newPtr);
    if (devId < 0) {
      THROW_EXCEPTION("DataBuffer::replaceSpecialBuffer: pointer is not a Vulkan allocation");
    }
  }
  clearSpecialWriteEvent();
  _specialBuffer = newPtr;
  _isOwnerSpecial = isOwner;
  _deviceId.store(devId);
  _specialDeviceId.store(devId);
}

// ─── allocateSpecial ───────────────────────────────────────────────────────────

void DataBuffer::allocateSpecial() {
  // Fast path: already allocated on the correct device.
  if (_specialBuffer != nullptr) {
    const int bufferDevice = resolveVulkanAllocationDevice(
        _specialBuffer, _specialDeviceId, _deviceId,
        "DataBuffer::allocateSpecial: invalid Vulkan allocation identity");
    if (isConstant) return;
    const int curDev = currentVulkanDeviceId();
    if (bufferDevice == curDev) return;
    // A frozen allocation cannot silently remain bound to the wrong device.
    // Callers must end the frozen lifetime before requesting migration.
    if (_frozenRefCount.load(std::memory_order_relaxed) > 0) {
      std::string message =
          "DataBuffer::allocateSpecial: frozen Vulkan allocation is bound to device ";
      message += std::to_string(_specialDeviceId.load());
      message += ", current device is ";
      message += std::to_string(curDev);
      THROW_EXCEPTION(message.c_str());
    }
    migrate();
    return;
  }

  throwIfFrozen("allocateSpecial");

  LongType lenBytes = getLenInBytes();
  if (lenBytes == 0) {
    lenBytes = _lenInBytes;
    if (lenBytes == 0) {
      // A device allocation must have a concrete byte extent.
      THROW_EXCEPTION("DataBuffer::allocateSpecial: length is 0, cannot allocate");
    }
  }

  int deviceId = currentVulkanDeviceId();

  void* ptr = sd::graph::VulkanMemoryPool::getInstance().allocate(
      deviceId, static_cast<VkDeviceSize>(lenBytes));

  if (ptr == nullptr) {
    std::string msg = "DataBuffer::allocateSpecial (Vulkan): allocation of ";
    msg += std::to_string(lenBytes);
    msg += " bytes failed on device ";
    msg += std::to_string(deviceId);
    THROW_EXCEPTION(msg.c_str());
  }

  _specialBuffer     = ptr;
  _specialAllocBytes = lenBytes;
  _isOwnerSpecial    = true;
  _deviceId.store(deviceId);
  _specialDeviceId.store(deviceId);

  DSP_DIAG(MEMORY, "Vulkan::allocateSpecial: %lld bytes @ ptr=%p dev=%d",
           (long long)lenBytes, ptr, deviceId);

#if defined(SD_GCC_FUNCTRACE)
  sd::array::DataBufferLifecycleTracker::getInstance().recordAllocation(
      _specialBuffer, lenBytes, _dataType,
      sd::array::BufferType::SPECIAL, this, _workspace != nullptr);
#endif

  tl_dspAllocBytes += lenBytes;
  tl_dspAllocCount++;
}

// ─── deleteSpecial ─────────────────────────────────────────────────────────────

void DataBuffer::deleteSpecial() {
  if (_isOwnerSpecial && _specialBuffer != nullptr && getLenInBytes() != 0) {
    const int devId = resolveVulkanAllocationDevice(
        _specialBuffer, _specialDeviceId, _deviceId,
        "DataBuffer::deleteSpecial: owned Vulkan allocation has invalid device identity");

    DSP_DIAG(MEMORY,
             "Vulkan::deleteSpecial: ordered release ptr=%p dev=%d len=%lld",
             _specialBuffer, devId, (long long)getLenInBytes());

#if defined(SD_GCC_FUNCTRACE)
    sd::array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
        _specialBuffer, sd::array::BufferType::SPECIAL);
#endif

    auto* releaseStream =
        sd::graph::VulkanExecutionStream::currentOrDefault(devId);
    if (releaseStream == nullptr || releaseStream->deviceId() != devId) {
      THROW_EXCEPTION(
          "DataBuffer::deleteSpecial: owning Vulkan stream is unavailable");
    }
    waitForSpecialWriteEvent(releaseStream);
    if (!releaseStream->retireAllocation(_specialBuffer)) {
      THROW_EXCEPTION(
          "DataBuffer::deleteSpecial: deferred Vulkan release failed");
    }

    tl_dspFreeBytes += getLenInBytes();
    tl_dspFreeCount++;
  }

  clearSpecialWriteEvent();
  _specialBuffer = nullptr;
  _specialDeviceId.store(-1);
  _deviceId.store(-1);
  _isOwnerSpecial = false;
}

// ─── freeGpuOnly / freeGpuOnStream ─────────────────────────────────────────────

void DataBuffer::freeGpuOnly() {
  throwIfFrozen("freeGpuOnly");
  deleteSpecial();
  deletePrimary();
  closed = true;
}

void DataBuffer::freeGpuOnStream(void* stream) {
  throwIfFrozen("freeGpuOnStream");
  if (_isOwnerSpecial && _specialBuffer != nullptr && getLenInBytes() != 0) {
    const int deviceId = resolveVulkanAllocationDevice(
        _specialBuffer, _specialDeviceId, _deviceId,
        "DataBuffer::freeGpuOnStream: invalid Vulkan allocation identity");
    auto* executionStream = stream == nullptr
        ? sd::graph::VulkanExecutionStream::defaultExecution(deviceId)
        : sd::graph::VulkanExecutionStream::fromOpaque(stream, false);
    if (executionStream == nullptr || executionStream->deviceId() != deviceId) {
      THROW_EXCEPTION("DataBuffer::freeGpuOnStream: stream belongs to another device");
    }
    waitForSpecialWriteEvent(executionStream);
    if (!executionStream->retireAllocation(_specialBuffer)) {
      THROW_EXCEPTION("DataBuffer::freeGpuOnStream: deferred Vulkan release failed");
    }
    tl_dspFreeBytes += getLenInBytes();
    tl_dspFreeCount++;
  }

  clearSpecialWriteEvent();
  _specialBuffer = nullptr;
  _specialDeviceId.store(-1);
  _deviceId.store(-1);
  _isOwnerSpecial = false;
  deletePrimary();
  closed = true;
}

// ─── syncToPrimary ─────────────────────────────────────────────────────────────

void DataBuffer::syncToPrimary(const LaunchContext* /*context*/, const bool forceSync) {
  if (_specialBuffer == nullptr || _lenInBytes == 0 || closed) return;
  if (isConstant && !forceSync) return;
  if (tl_graphExecutionActive && !forceSync) return;
  if (isPrimaryActual() && !forceSync) return;

  allocatePrimary();
  if (_primaryBuffer == nullptr) {
    THROW_EXCEPTION("DataBuffer::syncToPrimary: primary allocation failed");
  }

  const int deviceId = resolveVulkanAllocationDevice(
      _specialBuffer, _specialDeviceId, _deviceId,
      "DataBuffer::syncToPrimary: invalid Vulkan allocation identity");
  auto* copyStream = sd::graph::VulkanExecutionStream::defaultCopy(deviceId);
  if (copyStream == nullptr) {
    THROW_EXCEPTION("DataBuffer::syncToPrimary: Vulkan copy stream unavailable");
  }
  waitForSpecialWriteEvent(copyStream);

  auto startTime = std::chrono::high_resolution_clock::now();
  if (!sd::graph::VulkanMemoryPool::getInstance().copyDeviceToHost(
          _primaryBuffer, _specialBuffer,
          static_cast<VkDeviceSize>(getLenInBytes()))) {
    THROW_EXCEPTION("DataBuffer::syncToPrimary: Vulkan D2H transfer failed");
  }

  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - startTime).count();
  DSP_DIAG(MEMORY, "Vulkan::syncToPrimary: %lld bytes dev=%d dt=%.2f us",
           (long long)getLenInBytes(), deviceId, durationNs / 1000.0);
  readPrimary();
}

// ─── syncToSpecial ─────────────────────────────────────────────────────────────

void DataBuffer::syncToSpecial(const bool forceSync) {
  if (_primaryBuffer == nullptr || _lenInBytes == 0) return;
  if (isSpecialActual() && !forceSync) return;
  if (!forceSync && !isPrimaryActual() && _writeSpecial.load() > 0) return;
  if (tl_graphExecutionActive && !forceSync) return;

  allocateSpecial();
  if (_specialBuffer == nullptr) {
    THROW_EXCEPTION("DataBuffer::syncToSpecial: special allocation failed");
  }

  const int deviceId = resolveVulkanAllocationDevice(
      _specialBuffer, _specialDeviceId, _deviceId,
      "DataBuffer::syncToSpecial: invalid Vulkan allocation identity");
  auto* copyStream = sd::graph::VulkanExecutionStream::defaultCopy(deviceId);
  if (copyStream == nullptr) {
    THROW_EXCEPTION("DataBuffer::syncToSpecial: Vulkan copy stream unavailable");
  }
  waitForSpecialWriteEvent(copyStream);

  auto startTime = std::chrono::high_resolution_clock::now();
  if (!sd::graph::VulkanMemoryPool::getInstance().copyHostToDevice(
          _specialBuffer, _primaryBuffer,
          static_cast<VkDeviceSize>(getLenInBytes()))) {
    THROW_EXCEPTION("DataBuffer::syncToSpecial: Vulkan H2D transfer failed");
  }

  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - startTime).count();
  DSP_DIAG(MEMORY, "Vulkan::syncToSpecial: %lld bytes dev=%d dt=%.2f us",
           (long long)getLenInBytes(), deviceId, durationNs / 1000.0);
  // Match CUDA's dual-buffer state machine: H2D synchronization copies the
  // authoritative host value; it is not a device-side mutation.
  readSpecial();
}

// ─── allocateBuffers ──────────────────────────────────────────────────────────

void DataBuffer::allocateBuffers(const bool allocBoth) {
  // Match CUDA's device-backend contract: native arrays always own device
  // storage, while callers request host staging explicitly through allocBoth.
  allocateSpecial();
  if (allocBoth) allocatePrimary();
}

// ─── setSpecial ───────────────────────────────────────────────────────────────

void DataBuffer::setSpecial(void* special, const bool isOwnerSpecial) {
  int devId = -1;
  if (special != nullptr) {
    devId = sd::graph::VulkanMemoryPool::getInstance().getDeviceId(special);
    if (devId < 0) {
      THROW_EXCEPTION("DataBuffer::setSpecial: pointer is not a Vulkan allocation");
    }
  }
  deleteSpecial();
  _specialBuffer = special;
  _isOwnerSpecial = isOwnerSpecial;
  _deviceId.store(devId);
  _specialDeviceId.store(devId);
}

// ─── setToZeroBuffers ─────────────────────────────────────────────────────────

void DataBuffer::setToZeroBuffers(const bool both) {
  if (getLenInBytes() < 1 || _specialBuffer == nullptr) return;

  const int deviceId = resolveVulkanAllocationDevice(
      _specialBuffer, _specialDeviceId, _deviceId,
      "DataBuffer::setToZeroBuffers: invalid Vulkan allocation identity");

  if (!sd::graph::VulkanMemoryPool::getInstance().fill(
          _specialBuffer, 0,
          static_cast<VkDeviceSize>(getLenInBytes()))) {
    THROW_EXCEPTION("DataBuffer::setToZeroBuffers: Vulkan device fill failed");
  }
  writeSpecial();

  if (both && _primaryBuffer != nullptr) {
    std::memset(_primaryBuffer, 0, getLenInBytes());
    readPrimary();
  }

  DSP_DIAG(MEMORY, "Vulkan::setToZeroBuffers: %lld bytes dev=%d both=%d",
           (long long)getLenInBytes(), deviceId, both ? 1 : 0);
}

// ─── migrate ──────────────────────────────────────────────────────────────────

void DataBuffer::migrate() {
  // Migration is a direct Vulkan device-to-device operation. The host copy is
  // neither read nor written, so primary/special actuality remains unchanged.
  if (isConstant) return;
  throwIfFrozen("migrate");

  if (_specialBuffer == nullptr || getLenInBytes() == 0) return;

  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  const int newDev = currentVulkanDeviceId();
  const int oldDev = resolveVulkanAllocationDevice(
      _specialBuffer, _specialDeviceId, _deviceId,
      "DataBuffer::migrate: invalid Vulkan allocation identity");
  if (oldDev == newDev) return;

  std::string capabilityReason;
  if (!sd::graph::VulkanExecutionStream::isCrossDeviceCopySupported(
          oldDev, newDev, &capabilityReason)) {
    std::string message =
        "DataBuffer::migrate: Vulkan cross-device copy is unsupported: ";
    message += capabilityReason;
    THROW_EXCEPTION(message.c_str());
  }

  auto* targetStream =
      sd::graph::VulkanExecutionStream::currentOrDefault(newDev);
  if (targetStream == nullptr || targetStream->deviceId() != newDev) {
    THROW_EXCEPTION(
        "DataBuffer::migrate: destination Vulkan stream is unavailable");
  }

  const auto bytes = static_cast<VkDeviceSize>(getLenInBytes());
  void* newBuffer = pool.allocate(newDev, bytes);
  if (newBuffer == nullptr) {
    THROW_EXCEPTION(
        "DataBuffer::migrate: destination Vulkan allocation failed");
  }

  void* oldBuffer = _specialBuffer;
  const bool releaseOld = _isOwnerSpecial;
  if (!pool.copyDeviceToDeviceAsync(
          newBuffer, oldBuffer, bytes, targetStream)) {
    pool.freeImmediate(newBuffer);
    std::string message =
        "DataBuffer::migrate: Vulkan external-memory peer copy failed for device pair ";
    message += std::to_string(oldDev);
    message += " -> ";
    message += std::to_string(newDev);
    THROW_EXCEPTION(message.c_str());
  }

  if (releaseOld &&
      targetStream->enqueueHostCallback([oldBuffer]() {
        sd::graph::VulkanMemoryPool::getInstance().freeImmediate(oldBuffer);
      }) == 0) {
    // Error cleanup only: discard the submitted copy destination only after its
    // completion is proven. If synchronization fails, the pool retains the
    // allocation until device teardown rather than risking an in-flight free.
    if (targetStream->synchronize()) {
      pool.freeImmediate(newBuffer);
      THROW_EXCEPTION(
          "DataBuffer::migrate: failed to order source allocation release");
    }
    THROW_EXCEPTION(
        "DataBuffer::migrate: failed to order source allocation release and "
        "could not prove peer-copy completion");
  }

#if defined(SD_GCC_FUNCTRACE)
  sd::array::DataBufferLifecycleTracker::getInstance().recordAllocation(
      newBuffer, getLenInBytes(), _dataType,
      sd::array::BufferType::SPECIAL, this, _workspace != nullptr);
  if (releaseOld) {
    sd::array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
        oldBuffer, sd::array::BufferType::SPECIAL);
  }
#endif

  _specialBuffer = newBuffer;
  _specialAllocBytes = getLenInBytes();
  _isOwnerSpecial = true;
  _specialDeviceId.store(newDev);
  _deviceId.store(newDev);

  tl_dspAllocBytes += getLenInBytes();
  tl_dspAllocCount++;
  if (releaseOld) {
    tl_dspFreeBytes += getLenInBytes();
    tl_dspFreeCount++;
  }

  // Reuse the target-stream event as the completion proof for every later
  // consumer and for destruction of the newly owned allocation.
  recordSpecialWriteEvent(targetStream);

  DSP_DIAG(MEMORY,
           "Vulkan::migrate: external-memory peer copy %d -> %d ptr=%p "
           "newPtr=%p len=%lld",
           oldDev, newDev, oldBuffer, newBuffer,
           (long long)getLenInBytes());
}

// ─── Diagnostics / printing ────────────────────────────────────────────────────

void DataBuffer::printSpecialAllocationTraces() {
  // No-op on Vulkan (no StackTrace infra wired to pool).
}

void DataBuffer::showBufferLimited() {
  // No-op.
}

void DataBuffer::showCounters(const char* msg1, const char* msg2) {
  sd_debug("%s %s || primary %p special %p :: wP: %d wS: %d rP: %d rS: %d\n",
           msg1, msg2, _primaryBuffer, _specialBuffer,
           (int)_writePrimary.load(), (int)_writeSpecial.load(),
           (int)_readPrimary.load(), (int)_readSpecial.load());
}

// Must be defined before printHostDevice which uses BUILD_SINGLE_SELECTOR on it.
template <typename T>
void _printHostBuffer(DataBuffer* buffer, long offset) {
  sd::LongType len = buffer->getNumElements();
  auto buff = buffer->template primaryAsT<T>();
  sd::LongType limit = len;
  printf("[");
  sd::DataType dataType = buffer->getDataType();
  for (sd::LongType e = offset; e < limit; e++) {
    if (e > offset) printf(", ");
    if (dataType == sd::DataType::DOUBLE)
      printf("%.15f", (double)buff[e]);
    else if (dataType == sd::DataType::FLOAT32)
      printf("%.15f", (float)buff[e]);
    else if (dataType == sd::DataType::INT64 || dataType == sd::DataType::UINT64)
      printf("%lld", (long long)buff[e]);
    else if (dataType == sd::DataType::INT32 || dataType == sd::DataType::UINT32)
      printf("%d", (int)buff[e]);
    else if (dataType == sd::DataType::BOOL)
      printf("%s", (bool)buff[e] ? "true" : "false");
    else
      printf("%g", (double)buff[e]);
  }
  printf("]\n");
  fflush(stdout);
}

// Explicit instantiations so that BUILD_SINGLE_SELECTOR in printHostDevice
// can resolve _printHostBuffer<T> for every SD_COMMON_TYPES specialisation.
BUILD_SINGLE_TEMPLATE(SD_LIB_EXPORT void _printHostBuffer,
                      (sd::DataBuffer*, long),
                      SD_COMMON_TYPES);

void DataBuffer::printHostDevice(long offset) {
  auto xType = getDataType();
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer, (this, offset), SD_COMMON_TYPES);
}

void DataBuffer::printBufferDebug(const char* msg, sd::LongType offset, sd::LongType limit) {
  if (msg) sd_printf("%s:\n", msg);
  sd_printf("DataBuffer(Vulkan): DataType=%s, Length=%lld elements, DeviceId=%d\n",
            DataTypeUtils::asString(_dataType).c_str(), (long long)getNumElements(), deviceId());
  if (_primaryBuffer != nullptr) {
    sd_printf("Host buffer (@%p): first element = <see printHostDevice>\n", _primaryBuffer);
  } else {
    sd_printf("Host buffer: nullptr\n");
  }
  if (_specialBuffer != nullptr) {
    sd_printf("Device buffer (@%p): [Vulkan token — use syncToPrimary to read]\n", _specialBuffer);
  } else {
    sd_printf("Device buffer: nullptr\n");
  }
  sd_printf("Sync state: wP=%lld wS=%lld rP=%lld rS=%lld isPrimaryActual=%d isSpecialActual=%d\n",
            (long long)_writePrimary.load(), (long long)_writeSpecial.load(),
            (long long)_readPrimary.load(), (long long)_readSpecial.load(),
            isPrimaryActual() ? 1 : 0, isSpecialActual() ? 1 : 0);
}

// ─── Template instantiations ────────────────────────────────────────────────────

template <typename T>
void DataBuffer::printHostBufferContent(void* buffer, sd::LongType offset, sd::LongType length) {
  T* typedBuffer = reinterpret_cast<T*>(buffer);
  sd_printf("[ ");
  for (sd::LongType i = offset; i < offset + length; i++) {
    if (std::is_arithmetic<T>::value) {
      sd_printf("%g ", (double)typedBuffer[i]);
    } else {
      sd_printf("0x%x ", *reinterpret_cast<int*>(&typedBuffer[i]));
    }
  }
  sd_printf("]");
}
BUILD_SINGLE_TEMPLATE(SD_LIB_EXPORT void DataBuffer::printHostBufferContent,
                      (void* buffer, sd::LongType offset, sd::LongType length),
                      SD_COMMON_TYPES);

template <typename T>
void* DataBuffer::primaryAtOffset(const LongType offset) {
  if (_primaryBuffer == nullptr) return nullptr;
  T* type = reinterpret_cast<T*>(_primaryBuffer);
  return reinterpret_cast<void*>(type + offset);
}

template <typename T>
void* DataBuffer::specialAtOffset(const LongType offset) {
  if (_specialBuffer == nullptr) return nullptr;

  // Vulkan special pointers are stable allocation identities used to recover
  // VkBuffer/VkDeviceMemory records. DEVICE_LOCAL identities are never
  // arithmetic values; callers carry byte/element offsets in shape or
  // descriptor metadata.
  (void)offset;
  return _specialBuffer;
}

// Explicit instantiations for primaryAtOffset / specialAtOffset.
#define MAKE_OFFSET_INST(T)  \
  template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<T>(sd::LongType); \
  template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<T>(sd::LongType);

MAKE_OFFSET_INST(bool)
MAKE_OFFSET_INST(float16)
MAKE_OFFSET_INST(float8)
MAKE_OFFSET_INST(float8_e5m2)
MAKE_OFFSET_INST(bfloat16)
MAKE_OFFSET_INST(float)
MAKE_OFFSET_INST(double)
MAKE_OFFSET_INST(int8_t)
MAKE_OFFSET_INST(uint8_t)
MAKE_OFFSET_INST(int16_t)
MAKE_OFFSET_INST(int32_t)
MAKE_OFFSET_INST(sd::LongType)
MAKE_OFFSET_INST(uint16_t)
MAKE_OFFSET_INST(uint32_t)
MAKE_OFFSET_INST(uint64_t)

#undef MAKE_OFFSET_INST

// ─── expand ───────────────────────────────────────────────────────────────────

void DataBuffer::expand(const uint64_t size) {
  throwIfFrozen("expand");
  if (static_cast<LongType>(size) <= _lenInBytes) return;

  // Expand host (primary) buffer.
  int8_t* newBuffer = nullptr;
  size_t allocSize = size + (_workspace == nullptr ? static_cast<size_t>(HOST_ALLOC_PADDING) : 0);
  ALLOCATE(newBuffer, _workspace, allocSize, int8_t);
  if (_primaryBuffer != nullptr && _lenInBytes > 0) {
    std::memcpy(newBuffer, _primaryBuffer, _lenInBytes);
  }
  if (_workspace == nullptr) {
    uint64_t* canary = reinterpret_cast<uint64_t*>(newBuffer + size);
    for (size_t i = 0; i < (HOST_ALLOC_PADDING / sizeof(uint64_t)); i++) {
      canary[i] = sd::CanaryConstants::DATA_BUFFER_CANARY;
    }
  }
  if (_isOwnerPrimary) {
    auto ipb = reinterpret_cast<int8_t*>(_primaryBuffer);
    RELEASE(ipb, _workspace);
  }
  _primaryBuffer     = newBuffer;
  _primaryAllocBytes = allocSize;
  _isOwnerPrimary    = true;

  // Allocate a replacement device buffer after synchronously retiring the
  // old VkBuffer and its bound suballocation.
  if (_isOwnerSpecial && _specialBuffer != nullptr) {
    if (!sd::graph::VulkanMemoryPool::getInstance().freeSynchronized(
            _specialBuffer)) {
      THROW_EXCEPTION("DataBuffer::expand: synchronized Vulkan release failed");
    }
    _specialBuffer  = nullptr;
    _isOwnerSpecial = false;
  }
  _lenInBytes = size;
  allocateSpecial();

  // Sync new host data → device.
  syncToSpecial(/*forceSync=*/true);
}

// ─── copyBufferFrom / copyBufferFromHost ──────────────────────────────────────

void DataBuffer::copyBufferFrom(const DataBuffer& other,
                                size_t sizeToCopyinBytes,
                                const sd::LongType offsetThis,
                                const sd::LongType offsetOther) {
  if (other._primaryBuffer == nullptr && other._specialBuffer == nullptr) return;

  const VkDeviceSize dstOffset = checkedElementByteOffset(
      offsetThis, getLenInBytes(),
      DataTypeUtils::sizeOfElement(_dataType),
      "DataBuffer::copyBufferFrom: destination offset is out of bounds");
  const VkDeviceSize srcOffset = checkedElementByteOffset(
      offsetOther, other.getLenInBytes(),
      DataTypeUtils::sizeOfElement(other._dataType),
      "DataBuffer::copyBufferFrom: source offset is out of bounds");
  const VkDeviceSize dstRemaining =
      static_cast<VkDeviceSize>(getLenInBytes()) - dstOffset;
  const VkDeviceSize srcRemaining =
      static_cast<VkDeviceSize>(other.getLenInBytes()) - srcOffset;
  const VkDeviceSize copyBytes =
      sizeToCopyinBytes == 0
          ? std::min(dstRemaining, srcRemaining)
          : static_cast<VkDeviceSize>(sizeToCopyinBytes);
  if (copyBytes == 0) return;
  if (copyBytes > dstRemaining || copyBytes > srcRemaining) {
    THROW_EXCEPTION("DataBuffer::copyBufferFrom: copy range is out of bounds");
  }

  if (_specialBuffer == nullptr) allocateSpecial();
  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  const int dstDevice = resolveVulkanAllocationDevice(
      _specialBuffer, _specialDeviceId, _deviceId,
      "DataBuffer::copyBufferFrom: invalid destination Vulkan allocation identity");

  auto* copyStream =
      sd::graph::VulkanExecutionStream::currentOrDefault(dstDevice);
  if (copyStream == nullptr || copyStream->deviceId() != dstDevice) {
    THROW_EXCEPTION(
        "DataBuffer::copyBufferFrom: destination Vulkan stream is unavailable");
  }
  waitForSpecialWriteEvent(copyStream);

  if (other.isPrimaryActual()) {
    if (other._primaryBuffer == nullptr ||
        !pool.copyHostToDeviceAsync(
            _specialBuffer, other._primaryBuffer, copyBytes, copyStream,
            dstOffset, srcOffset)) {
      THROW_EXCEPTION(
          "DataBuffer::copyBufferFrom: Vulkan host-to-device copy failed");
    }
    other.readPrimary();
  } else {
    if (other._specialBuffer == nullptr) {
      THROW_EXCEPTION(
          "DataBuffer::copyBufferFrom: source has no authoritative Vulkan storage");
    }
    const int srcDevice = validateVulkanAllocationDevice(
        other._specialBuffer, other._specialDeviceId,
        "DataBuffer::copyBufferFrom: invalid source Vulkan allocation identity");

    auto* sourceStream =
        sd::graph::VulkanExecutionStream::currentOrDefault(srcDevice);
    if (sourceStream == nullptr || sourceStream->deviceId() != srcDevice) {
      THROW_EXCEPTION(
          "DataBuffer::copyBufferFrom: source Vulkan stream is unavailable");
    }
    other.waitForSpecialWriteEvent(sourceStream);

    if (!pool.copyDeviceToDeviceAsync(
            _specialBuffer, other._specialBuffer, copyBytes, copyStream,
            dstOffset, srcOffset)) {
      THROW_EXCEPTION(
          "DataBuffer::copyBufferFrom: direct Vulkan device copy failed");
    }
    other.readSpecial();
  }

  recordSpecialWriteEvent(copyStream);
  writeSpecial();
}

void DataBuffer::copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes,
                                    const sd::LongType offsetThis,
                                    const sd::LongType offsetHostBuffer) {
  if (hostBuffer == nullptr) return;

  const size_t elementSize = DataTypeUtils::sizeOfElement(_dataType);
  const VkDeviceSize dstOffset = checkedElementByteOffset(
      offsetThis, getLenInBytes(), elementSize,
      "DataBuffer::copyBufferFromHost: destination offset is out of bounds");
  const VkDeviceSize srcOffset = checkedHostByteOffset(
      offsetHostBuffer, elementSize,
      "DataBuffer::copyBufferFromHost: host offset is out of bounds");
  const VkDeviceSize remaining =
      static_cast<VkDeviceSize>(getLenInBytes()) - dstOffset;
  const VkDeviceSize copyBytes =
      sizeToCopyinBytes == 0
          ? remaining
          : static_cast<VkDeviceSize>(sizeToCopyinBytes);
  if (copyBytes == 0) return;
  if (copyBytes > remaining) {
    THROW_EXCEPTION(
        "DataBuffer::copyBufferFromHost: copy range is out of bounds");
  }

  if (_specialBuffer == nullptr) allocateSpecial();
  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  const int dstDevice = resolveVulkanAllocationDevice(
      _specialBuffer, _specialDeviceId, _deviceId,
      "DataBuffer::copyBufferFromHost: invalid Vulkan allocation identity");

  auto* copyStream =
      sd::graph::VulkanExecutionStream::currentOrDefault(dstDevice);
  if (copyStream == nullptr || copyStream->deviceId() != dstDevice) {
    THROW_EXCEPTION(
        "DataBuffer::copyBufferFromHost: Vulkan stream is unavailable");
  }
  waitForSpecialWriteEvent(copyStream);

  if (!pool.copyHostToDeviceAsync(
          _specialBuffer, hostBuffer, copyBytes, copyStream,
          dstOffset, srcOffset)) {
    THROW_EXCEPTION(
        "DataBuffer::copyBufferFromHost: Vulkan host-to-device copy failed");
  }

  recordSpecialWriteEvent(copyStream);
  writeSpecial();
}

// ─── memcpy (static) ──────────────────────────────────────────────────────────

template <typename T>
void memcpyWithT(DataBuffer* dst, DataBuffer* src, sd::LongType startingOffset,
                 sd::LongType dstOffset, sd::LongType n) {
  const auto sizeOfElement =
      static_cast<sd::LongType>(DataTypeUtils::sizeOfElement(src->getDataType()));
  sd::LongType srcAvailable =
      src->getLenInBytes() - startingOffset * sizeOfElement;
  sd::LongType dstAvailable =
      dst->getLenInBytes() - dstOffset * sizeOfElement;
  sd::LongType copyBytes =
      n > 0 ? n * sizeOfElement
            : (srcAvailable < dstAvailable ? srcAvailable : dstAvailable);
  if (copyBytes > srcAvailable) copyBytes = srcAvailable;
  if (copyBytes > dstAvailable) copyBytes = dstAvailable;
  if (copyBytes <= 0) return;

  // DataBuffer::memcpy is the device-side copy primitive on accelerator
  // backends.  copyBufferFrom preserves Vulkan stream ordering, chooses H2D
  // or D2D from the source actuality counters, records the destination write
  // event, and marks the destination special buffer authoritative.
  dst->copyBufferFrom(*src, static_cast<size_t>(copyBytes), dstOffset,
                      startingOffset);
}

// Explicit instantiations for memcpyWithT so BUILD_SINGLE_SELECTOR can resolve them.
BUILD_SINGLE_TEMPLATE(SD_LIB_EXPORT void memcpyWithT,
                      (sd::DataBuffer*, sd::DataBuffer*, sd::LongType, sd::LongType, sd::LongType),
                      SD_COMMON_TYPES);

void DataBuffer::memcpy(DataBuffer* dst, DataBuffer* src,
                        sd::LongType startingOffset, sd::LongType dstOffset, sd::LongType n) {
  BUILD_SINGLE_SELECTOR(src->getDataType(), memcpyWithT,
                        (dst, src, startingOffset, dstOffset, n), SD_COMMON_TYPES);
}

// ─── DataBuffer::dup ──────────────────────────────────────────────────────────

DataBuffer DataBuffer::dup() {
  DataBuffer result;
  result._dataType   = _dataType;
  result._lenInBytes = _lenInBytes;
  result._primaryBuffer  = nullptr;
  result._specialBuffer  = nullptr;
  result._isOwnerPrimary = false;
  result._isOwnerSpecial = false;
  result.allocateBuffers(true);
  result.copyCounters(*this);
  result.copyBufferFrom(*this);
  return result;
}

}  // namespace sd

#endif  // SD_VULKAN
