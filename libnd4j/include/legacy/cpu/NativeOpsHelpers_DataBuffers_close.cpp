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

#include <array/InteropDataBuffer.h>
#include <helpers/logger.h>
#include <system/Environment.h>
#include <atomic>
#include <thread>
#include <legacy/NativeOps.h>

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif

// These counters are defined in NativeOpsHelpers_DataBuffers.cpp
extern std::atomic<size_t> g_dataBufferCount;
extern std::atomic<size_t> g_dataBufferBytes;

using OpaqueDataBuffer = sd::InteropDataBuffer;

// CPU stubs for dbClose diagnostics (only meaningful on CUDA)
void dbCloseGetDiagnostics(sd::LongType* outStats) {
  for (int i = 0; i < 9; i++) outStats[i] = 0;
}

void dbCloseResetDiagnostics() {
  // no-op on CPU
}

/**
 * CPU implementation of dbClose - closes and frees a DataBuffer.
 * On CPU, no device synchronization is needed.
 *
 * THREAD SAFETY: This function uses atomic compare-and-swap on _closed
 * to ensure only ONE thread can close a buffer, preventing double-free
 * in multi-threaded environments like Spring DeallocatorService.
 */
void dbClose(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbClose: dataBuffer is null");

  if(sd::Environment::getInstance().isLogNativeNDArrayCreation()) {
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    sd_printf("dbClose: called on buffer at %p, isConstant=%d, cachedLen=%lld, cachedPrimary=%p, cachedSpecial=%p, threadId=%llu\n",
              dataBuffer, dataBuffer->isConstant.load() ? 1 : 0, (long long)dataBuffer->_cachedLenInBytes,
              dataBuffer->_cachedPrimaryPtr, dataBuffer->_cachedSpecialPtr,
              (unsigned long long)tid);
    fflush(stdout);
  }

  // Check constant flag FIRST (public field, safe to access)
  // Constant buffers should never be freed
  if(dataBuffer->isConstant.load(std::memory_order_acquire)) {
    if(sd::Environment::getInstance().isVerbose() || sd::Environment::getInstance().isLogNativeNDArrayCreation()) {
      sd_printf("dbClose: skipping constant buffer at %p\n", dataBuffer);
      fflush(stdout);
    }
    return;
  }

  if(!dataBuffer->tryClose()) {
    // Another thread already closed this buffer - do nothing
    if(sd::Environment::getInstance().isVerbose() || sd::Environment::getInstance().isLogNativeNDArrayCreation()) {
      sd_printf("dbClose: buffer at %p already closed by another thread\n", dataBuffer);
      fflush(stdout);
    }
    return;
  }

  // From here on, we are the ONLY thread that will execute this code for this buffer
  // because tryClose() succeeded (atomically set _closed from false to true)

  // Check if we even have a DataBuffer pointer
  if(!dataBuffer->hasValidDataBuffer()) {
    // No DataBuffer to delete, but we already marked as closed via tryClose()
    return;
  }

  // If we don't own it, don't close it
  if(!dataBuffer->isOwner()) {
    return;
  }

  // Track deallocation using cached size - DO NOT touch the DataBuffer as it may be freed
  // Use the cached size from InteropDataBuffer instead of accessing potentially freed memory
  size_t bytes = dataBuffer->_cachedLenInBytes;
  g_dataBufferCount.fetch_sub(1, std::memory_order_relaxed);
  g_dataBufferBytes.fetch_sub(bytes, std::memory_order_relaxed);

  if(sd::Environment::getInstance().isVerbose()) {
    sd_printf("dbClose: deallocating buffer at %p, count=%zu, total_bytes=%zu, freed_bytes=%zu\n",
              dataBuffer, g_dataBufferCount.load(), g_dataBufferBytes.load(), bytes);
  }

#if defined(SD_GCC_FUNCTRACE)
  // Record deallocation using cached pointers (safe even if DataBuffer is freed)
  if(dataBuffer->_cachedPrimaryPtr != nullptr) {
    sd::array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
        dataBuffer->_cachedPrimaryPtr, sd::array::BufferType::PRIMARY);
  }
  if(dataBuffer->_cachedSpecialPtr != nullptr) {
    sd::array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
        dataBuffer->_cachedSpecialPtr, sd::array::BufferType::SPECIAL);
  }
#endif

  sd::DataBuffer* db = dataBuffer->getDataBufferDirect();

  dataBuffer->waitForNoReaders();

  // Invalidate the DataBuffer pointer BEFORE deleting to prevent concurrent access
  // Note: We already marked _closed = true via tryClose() at the start
  dataBuffer->invalidateDataBuffer();

  // Delete the DataBuffer if we have one and we own it
  // This is safe because:
  // 1. We passed the isOwner() check above
  // 2. tryClose() ensures only ONE thread executes this code
  // 3. We waited for all in-flight accesses to complete
  if(db != nullptr) {
    if(sd::Environment::getInstance().isLogNativeNDArrayCreation()) {
      sd_printf("dbClose: about to delete DataBuffer %p (primary=%p, special=%p, lenInBytes=%lld, isConstant=%d) for InteropDataBuffer %p\n",
                db, db->primary(), db->special(), (long long)db->getLenInBytes(),
                db->isConstant ? 1 : 0, dataBuffer);
      fflush(stdout);
    }
    delete db;
    if(sd::Environment::getInstance().isLogNativeNDArrayCreation()) {
      sd_printf("dbClose: successfully deleted DataBuffer for InteropDataBuffer %p\n", dataBuffer);
      fflush(stdout);
    }
  }
}

void dbFreeBuffersOnly(OpaqueDataBuffer *dataBuffer) {
  // CPU implementation: uses freeGpuOnly() which on CPU just abandons the buffer.
  // On CUDA, this frees GPU memory via deleteSpecial() and abandons host memory.
  if (dataBuffer == nullptr) return;
  if (dataBuffer->isConstant.load(std::memory_order_acquire)) return;
  if (!dataBuffer->tryClose()) return;
  if (!dataBuffer->hasValidDataBuffer()) return;
  if (!dataBuffer->isOwner()) return;

  size_t bytes = dataBuffer->_cachedLenInBytes;
  g_dataBufferCount.fetch_sub(1, std::memory_order_relaxed);
  g_dataBufferBytes.fetch_sub(bytes, std::memory_order_relaxed);

  sd::DataBuffer* db = dataBuffer->getDataBufferDirect();
  if (db == nullptr) return;

  db->freeGpuOnly();
  dataBuffer->invalidateDataBuffer();
  delete db;
}

void dbFreeBuffersOnStream(OpaqueDataBuffer *dataBuffer, void *stream) {
  // CPU: stream parameter is ignored, just delegates to dbFreeBuffersOnly
  dbFreeBuffersOnly(dataBuffer);
}

bool dbIsOwner(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) return false;
  return dataBuffer->isOwner();
}
