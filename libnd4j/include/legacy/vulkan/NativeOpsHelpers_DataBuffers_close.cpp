/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/InteropDataBuffer.h>
#include <legacy/NativeOps.h>

#include <atomic>
#include <cstdint>

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif


extern std::atomic<size_t> g_dataBufferCount;
extern std::atomic<size_t> g_dataBufferBytes;

namespace {

std::atomic<int64_t> g_dbCloseTotal{0};
std::atomic<int64_t> g_dbCloseNull{0};
std::atomic<int64_t> g_dbCloseConstant{0};
std::atomic<int64_t> g_dbCloseAlreadyClosed{0};
std::atomic<int64_t> g_dbCloseNoDataBuffer{0};
std::atomic<int64_t> g_dbCloseNotOwner{0};
std::atomic<int64_t> g_dbCloseDeviceError{0};
std::atomic<int64_t> g_dbCloseDeleted{0};
std::atomic<int64_t> g_dbCloseFreedBytes{0};

void recordSuccessfulFree(size_t bytes) {
  g_dataBufferCount.fetch_sub(1, std::memory_order_relaxed);
  g_dataBufferBytes.fetch_sub(bytes, std::memory_order_relaxed);
  g_dbCloseDeleted.fetch_add(1, std::memory_order_relaxed);
  g_dbCloseFreedBytes.fetch_add(static_cast<int64_t>(bytes),
                               std::memory_order_relaxed);
}

void restoreOpenAfterReleaseFailure(OpaqueDataBuffer* dataBuffer) {
  // tryClose() is the ownership claim for this close attempt. If native
  // retirement throws, the InteropDataBuffer still owns its DataBuffer and
  // must become accessible/retryable again. CAS avoids overwriting a state
  // transition not owned by this attempt.
  bool expected = true;
  dataBuffer->_closed.compare_exchange_strong(
      expected, false, std::memory_order_acq_rel, std::memory_order_acquire);
}

bool claimOwnedBuffer(OpaqueDataBuffer* dataBuffer, sd::DataBuffer*& buffer,
                      size_t& bytes) {
  if (dataBuffer == nullptr) return false;
  if (dataBuffer->isConstant.load(std::memory_order_acquire)) return false;
  if (!dataBuffer->tryClose()) return false;
  if (!dataBuffer->hasValidDataBuffer()) return false;
  if (!dataBuffer->isOwner()) return false;

  buffer = dataBuffer->getDataBufferDirect();
  if (buffer == nullptr) return false;
  bytes = dataBuffer->_cachedLenInBytes;
  dataBuffer->waitForNoReaders();
  return true;
}

}  // namespace

void dbCloseGetDiagnostics(sd::LongType* outStats) {
  if (outStats == nullptr) return;
  outStats[0] = g_dbCloseTotal.load(std::memory_order_relaxed);
  outStats[1] = g_dbCloseNull.load(std::memory_order_relaxed);
  outStats[2] = g_dbCloseConstant.load(std::memory_order_relaxed);
  outStats[3] = g_dbCloseAlreadyClosed.load(std::memory_order_relaxed);
  outStats[4] = g_dbCloseNoDataBuffer.load(std::memory_order_relaxed);
  outStats[5] = g_dbCloseNotOwner.load(std::memory_order_relaxed);
  outStats[6] = g_dbCloseDeviceError.load(std::memory_order_relaxed);
  outStats[7] = g_dbCloseDeleted.load(std::memory_order_relaxed);
  outStats[8] = g_dbCloseFreedBytes.load(std::memory_order_relaxed);
}

void dbCloseResetDiagnostics() {
  g_dbCloseTotal.store(0, std::memory_order_relaxed);
  g_dbCloseNull.store(0, std::memory_order_relaxed);
  g_dbCloseConstant.store(0, std::memory_order_relaxed);
  g_dbCloseAlreadyClosed.store(0, std::memory_order_relaxed);
  g_dbCloseNoDataBuffer.store(0, std::memory_order_relaxed);
  g_dbCloseNotOwner.store(0, std::memory_order_relaxed);
  g_dbCloseDeviceError.store(0, std::memory_order_relaxed);
  g_dbCloseDeleted.store(0, std::memory_order_relaxed);
  g_dbCloseFreedBytes.store(0, std::memory_order_relaxed);
}

void dbClose(OpaqueDataBuffer* dataBuffer) {
  g_dbCloseTotal.fetch_add(1, std::memory_order_relaxed);

  if (dataBuffer == nullptr) {
    g_dbCloseNull.fetch_add(1, std::memory_order_relaxed);
    THROW_EXCEPTION("dbClose: dataBuffer is null");
  }
  if (dataBuffer->isConstant.load(std::memory_order_acquire)) {
    g_dbCloseConstant.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (!dataBuffer->tryClose()) {
    g_dbCloseAlreadyClosed.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (!dataBuffer->hasValidDataBuffer()) {
    g_dbCloseNoDataBuffer.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (!dataBuffer->isOwner()) {
    g_dbCloseNotOwner.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  sd::DataBuffer* buffer = dataBuffer->getDataBufferDirect();
  if (buffer == nullptr) {
    g_dbCloseNoDataBuffer.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  const size_t bytes = dataBuffer->_cachedLenInBytes;
  dataBuffer->waitForNoReaders();

#if defined(SD_GCC_FUNCTRACE)
  if (dataBuffer->_cachedPrimaryPtr != nullptr) {
    sd::array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
        dataBuffer->_cachedPrimaryPtr, sd::array::BufferType::PRIMARY);
  }
#endif

  try {
    // The public Vulkan lifecycle entry resolves the owning device from
    // VulkanMemoryPool, waits for its write event, retires the allocation in
    // queue order, and releases any primary storage before object deletion.
    buffer->freeGpuOnly();
  } catch (...) {
    g_dbCloseDeviceError.fetch_add(1, std::memory_order_relaxed);
    restoreOpenAfterReleaseFailure(dataBuffer);
    throw;
  }

  dataBuffer->invalidateDataBuffer();
  delete buffer;
  recordSuccessfulFree(bytes);
}

void dbFreeBuffersOnly(OpaqueDataBuffer* dataBuffer) {
  sd::DataBuffer* buffer = nullptr;
  size_t bytes = 0;
  if (!claimOwnedBuffer(dataBuffer, buffer, bytes)) return;

  try {
    buffer->freeGpuOnly();
  } catch (...) {
    g_dbCloseDeviceError.fetch_add(1, std::memory_order_relaxed);
    restoreOpenAfterReleaseFailure(dataBuffer);
    throw;
  }

  dataBuffer->invalidateDataBuffer();
  delete buffer;
  recordSuccessfulFree(bytes);
}

void dbFreeBuffersOnStream(OpaqueDataBuffer* dataBuffer, void* stream) {
  sd::DataBuffer* buffer = nullptr;
  size_t bytes = 0;
  if (!claimOwnedBuffer(dataBuffer, buffer, bytes)) return;

  try {
    // Vulkan validates that the opaque stream belongs to the allocation's
    // owning device and retires the allocation in that stream's queue order.
    buffer->freeGpuOnStream(stream);
  } catch (...) {
    g_dbCloseDeviceError.fetch_add(1, std::memory_order_relaxed);
    restoreOpenAfterReleaseFailure(dataBuffer);
    throw;
  }

  dataBuffer->invalidateDataBuffer();
  delete buffer;
  recordSuccessfulFree(bytes);
}

bool dbIsOwner(OpaqueDataBuffer* dataBuffer) {
  return dataBuffer != nullptr && dataBuffer->isOwner();
}


#endif  // SD_VULKAN && HAVE_VULKAN
