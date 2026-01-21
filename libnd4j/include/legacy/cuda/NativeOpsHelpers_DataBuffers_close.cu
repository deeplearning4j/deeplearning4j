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
#include <cuda_runtime.h>
#include <atomic>
#include <legacy/NativeOps.h>

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif

// These counters are defined in NativeOpsHelpers_DataBuffers.cpp
extern std::atomic<size_t> g_dataBufferCount;
extern std::atomic<size_t> g_dataBufferBytes;


void dbClose(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbClose: dataBuffer is null");

  // Check constant flag FIRST (public field, safe to access)
  // Constant buffers should never be freed
  if(dataBuffer->isConstant.load(std::memory_order_acquire)) {
    if(sd::Environment::getInstance().isVerbose()) {
      sd_printf("dbClose: skipping constant buffer at %p\n", dataBuffer);
    }
    return;
  }

  if(!dataBuffer->tryClose()) {
    // Another thread already closed this buffer - do nothing
    if(sd::Environment::getInstance().isVerbose()) {
      sd_printf("dbClose: buffer at %p already closed by another thread\n", dataBuffer);
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

  // Get the DataBuffer before marking closed
  sd::DataBuffer* db = dataBuffer->getDataBufferDirect();

  int bufferDeviceId = dataBuffer->deviceId();

  cudaError_t pendingErr = cudaGetLastError();
  if (pendingErr != cudaSuccess) {
    sd_debug("dbClose: Cleared pending CUDA error: %s\n", cudaGetErrorString(pendingErr));
  }

  int currentDevice = 0;
  cudaGetDevice(&currentDevice);

  cudaError_t setDevErr = cudaSetDevice(bufferDeviceId);
  if (setDevErr != cudaSuccess) {
    // Can't set device - this is CRITICAL. We cannot safely free this buffer
    // because we can't sync the correct device. This would cause use-after-free
    // corruption if async CUDA operations are still using the buffer.
    cudaGetLastError();  // Clear the error
    sd_printf("dbClose: CRITICAL - Failed to set device %d: %s. Cannot safely free buffer, deferring.\n",
              bufferDeviceId, cudaGetErrorString(setDevErr));
    // Do NOT delete the buffer - it's safer to leak than to corrupt memory
    // Note: _closed was already set to true via tryClose() at the start
    return;
  }

  dataBuffer->waitForNoReaders();

  cudaError_t syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    // Sync failed - this is also CRITICAL. We cannot safely free this buffer.
    cudaGetLastError();  // Clear the error
    sd_printf("dbClose: CRITICAL - cudaDeviceSynchronize failed on device %d: %s. Cannot safely free buffer, deferring.\n",
              bufferDeviceId, cudaGetErrorString(syncErr));
    // Restore original device
    cudaSetDevice(currentDevice);
    // Do NOT delete the buffer - it's safer to leak than to corrupt memory
    // Note: _closed was already set to true via tryClose() at the start
    return;
  }

  // Invalidate the DataBuffer pointer BEFORE deleting to prevent concurrent access
  // Note: _closed was already set to true via tryClose() at the start
  dataBuffer->invalidateDataBuffer();

  // Delete the DataBuffer if we have one and we own it
  // This is safe because:
  // 1. We passed the isOwner() check above
  // 2. tryClose() ensures only ONE thread executes this code
  // 3. We waited for all in-flight accesses to complete
  // 4. We synchronized the CUDA device before freeing
  if(db != nullptr) {
    delete db;
  }

  // Restore original device context if we switched devices
  if (currentDevice != bufferDeviceId) {
    cudaSetDevice(currentDevice);
  }
}
