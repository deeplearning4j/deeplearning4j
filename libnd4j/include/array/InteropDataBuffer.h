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

#include <array/DataBuffer.h>
#include <array/DataType.h>
#include <memory/Workspace.h>
#include <memory/MultiBackendWorkspace.h>
#include <system/common.h>

#include <memory>
#include <atomic>
#include <thread>
#include <chrono>

#ifndef LIBND4J_INTEROPDATABUFFER_H
#define LIBND4J_INTEROPDATABUFFER_H

namespace sd {

// Forward declaration
class InteropDataBuffer;

/**
 * RAII guard for safe buffer pointer access.
 *
 * This guard ensures that the InteropDataBuffer's in-use count is properly
 * managed, preventing use-after-free when dbClose() runs concurrently.
 *
 * Usage:
 *   BufferAccessGuard guard(buffer);
 *   if (guard.isValid()) {
 *     void* ptr = guard.primaryPtr();
 *     // Use ptr safely - buffer cannot be freed while guard exists
 *   }
 *   // Guard destructor releases access, allowing dbClose() to proceed
 *
 * This fixes the race condition where primary()/special() would release
 * access before returning the pointer, allowing dbClose() to free the
 * buffer while the caller still holds the pointer.
 */
class SD_LIB_EXPORT BufferAccessGuard {
 private:
  const InteropDataBuffer* _buffer;
  void* _primaryPtr;
  void* _specialPtr;
  bool _valid;

 public:
  // Acquire access and cache pointers
  explicit BufferAccessGuard(const InteropDataBuffer* buffer);

  // Release access on destruction
  ~BufferAccessGuard();

  // Prevent copy - guard must be unique owner of access
  BufferAccessGuard(const BufferAccessGuard&) = delete;
  BufferAccessGuard& operator=(const BufferAccessGuard&) = delete;

  // Allow move
  BufferAccessGuard(BufferAccessGuard&& other) noexcept;
  BufferAccessGuard& operator=(BufferAccessGuard&& other) noexcept;

  // Check if access was successfully acquired
  bool isValid() const { return _valid; }

  // Get cached pointers (valid only while guard exists)
  void* primaryPtr() const { return _primaryPtr; }
  void* specialPtr() const { return _specialPtr; }

  // Explicit release (guard becomes invalid after this)
  void release();
};

// Magic number for detecting use-after-free in InteropDataBuffer
// This specific value (0xAB12CD34EF56) is chosen to be unlikely to appear naturally
// and is validated in primary()/special() to catch dangling pointer access
constexpr uint64_t INTEROP_BUFFER_MAGIC = 0xAB12CD34EF56ULL;
constexpr uint64_t INTEROP_BUFFER_FREED = 0xDEADBEEFCAFEULL;

/**
 * This class is a wrapper for DataBuffer, suitable for sharing DataBuffer between front-end and back-end languages
 */
class SD_LIB_EXPORT InteropDataBuffer {
  // Allow BufferAccessGuard to access internal pointers without releasing
  friend class BufferAccessGuard;

 private:
  uint64_t _magic = INTEROP_BUFFER_MAGIC;
  std::atomic<DataBuffer*> _dataBuffer{nullptr};
  bool owner;
  DataType _dataType = DataType::UNKNOWN;
  memory::Workspace* _workspace = nullptr;  // Workspace backing this buffer (null = heap-allocated)
  memory::MultiBackendWorkspace* _multiBackendWorkspace = nullptr;  // Multi-device workspace for coherence tracking
 public:
  size_t _cachedLenInBytes = 0;
  void* _cachedPrimaryPtr = nullptr;  // Cached for deallocation tracking
  void* _cachedSpecialPtr = nullptr;  // Cached for deallocation tracking
  std::atomic<bool> _closed{false};
  std::atomic<bool> isConstant{false};
  mutable std::atomic<int32_t> _inUseCount{0};

  // Increment in-use count - returns false if buffer is closed (caller should not proceed)
  // Exception: constant buffers can always be accessed because their data is never freed.
  bool acquireAccess() const {
    // First increment the counter
    _inUseCount.fetch_add(1, std::memory_order_acq_rel);

    // Constant buffers are always accessible - their data is never freed
    if (isConstant.load(std::memory_order_acquire)) {
      return true;
    }

    // Then check if closed - if so, decrement and return false
    if (_closed.load(std::memory_order_acquire)) {
      _inUseCount.fetch_sub(1, std::memory_order_release);
      return false;
    }
    return true;
  }

  // Decrement in-use count
  void releaseAccess() const {
    _inUseCount.fetch_sub(1, std::memory_order_release);
  }

  // Wait for all readers to finish (used before deletion)
  void waitForNoReaders() const {
    // Spin-wait for in-use count to reach 0
    // Use exponential backoff to avoid busy-waiting
    int spinCount = 0;
    while (_inUseCount.load(std::memory_order_acquire) > 0) {
      if (spinCount < 10) {
        // Busy spin for first few iterations
        spinCount++;
      } else if (spinCount < 20) {
        // Yield to other threads
        std::this_thread::yield();
        spinCount++;
      } else {
        // Sleep briefly to avoid CPU burn
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
  }

  // Get current in-use count (for debugging)
  int32_t getInUseCount() const {
    return _inUseCount.load(std::memory_order_acquire);
  }

  // Validate the magic number - returns true if this object is valid
  // Constant buffers remain valid even after _closed is set, because
  // their data is never freed and their pointer is never nulled.
  bool isValid() const {
    bool bufferIsConstant = isConstant.load(std::memory_order_acquire);
    bool hasMagic = _magic == INTEROP_BUFFER_MAGIC;
    bool notClosed = !_closed.load(std::memory_order_acquire);

    // Constant buffers are valid as long as magic is intact
    // Non-constant buffers must also not be closed
    return hasMagic && (bufferIsConstant || notClosed);
  }

  // Atomically check if closed and set to closed if not already
  // Returns true if THIS call closed the buffer (i.e., it was open before)
  // Returns false if already closed (another thread won the race)
  bool tryClose() {
    bool expected = false;
    return _closed.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
  }

  // Check if the buffer is closed (thread-safe read)
  bool isClosed() const {
    return _closed.load(std::memory_order_acquire);
  }

  InteropDataBuffer(InteropDataBuffer *dataBuffer, uint64_t length);
  InteropDataBuffer(DataBuffer * databuffer);
  InteropDataBuffer(size_t lenInBytes, DataType dtype, bool allocateBoth);
  ~InteropDataBuffer() {
    bool weOwnClose = tryClose();

    // Check constant flag - constant buffers (like shape info) should never have
    // their DataBuffer pointer nulled, as other code may still reference them.
    bool bufferIsConstant = isConstant.load(std::memory_order_acquire);

    // Set magic to freed pattern to detect use-after-free
    // ONLY set this for non-constant buffers, constant buffers remain valid
    if (!bufferIsConstant) {
      _magic = INTEROP_BUFFER_FREED;
    }

    // Load the pointer atomically before checking/deleting
    DataBuffer* db = _dataBuffer.load(std::memory_order_acquire);

    // Only delete if:
    // 1. WE won the close race (tryClose() returned true)
    // 2. We own the DataBuffer
    // 3. It's not marked constant
    // 4. The pointer is not null
    // If dbClose() won the close race, it will handle deletion after waiting
    // for readers and syncing the device.
    if (weOwnClose && owner && !bufferIsConstant && db != nullptr) {
      delete db;
    }

    if (!bufferIsConstant) {
      _dataBuffer.store(nullptr, std::memory_order_release);
    }
  }

#ifndef __JAVACPP_HACK__
  DataBuffer * getDataBuffer() const;
  DataBuffer * dataBuffer();
  // In InteropDataBuffer class, add these public methods:
  // Check if the underlying DataBuffer pointer is still valid
  // by checking if it's non-null (thread-safe atomic read)
  bool hasValidDataBuffer() const {
    return _dataBuffer.load(std::memory_order_acquire) != nullptr;
  }

  // Get the DataBuffer pointer directly without validation (thread-safe atomic read)
  // Note: The returned pointer may become invalid if another thread closes the buffer
  DataBuffer* getDataBufferDirect() const {
    return _dataBuffer.load(std::memory_order_acquire);
  }

  // Mark the DataBuffer as invalid (set to null) - thread-safe atomic write
  // Uses release ordering to ensure all prior writes are visible to threads
  // that subsequently read nullptr
  void invalidateDataBuffer() {
    _dataBuffer.store(nullptr, std::memory_order_release);
  }

  bool isOwner() const { return owner; }

  // Safe accessor that doesn't touch other fields (thread-safe atomic read)
  DataBuffer* getDataBufferUnsafe() const {
    return _dataBuffer.load(std::memory_order_acquire);
  }
#endif

  void printDbAllocationTrace();

  void *primary() const;
  void *special() const;

  /**
   * Create a BufferAccessGuard for safe pointer access.
   *
   * This is the RECOMMENDED way to access buffer pointers when there's
   * any possibility of concurrent deallocation (e.g., from DeallocatorService).
   *
   * The guard acquires access, caches the pointers, and releases access
   * only when the guard is destroyed. This prevents the race condition
   * where primary()/special() release access before the caller uses the pointer.
   *
   * Usage:
   *   BufferAccessGuard guard = buffer->acquireGuard();
   *   if (guard.isValid()) {
   *     void* ptr = guard.primaryPtr();
   *     // Use ptr safely
   *   }
   *   // Guard destructor releases access
   */
  BufferAccessGuard acquireGuard() const;

 private:
  // Internal methods for BufferAccessGuard to use - do NOT release access
  void* primaryNoRelease() const;
  void* specialNoRelease() const;

 public:

  void markConstant(bool reallyConstant) {
    isConstant.store(reallyConstant, std::memory_order_release);
    dataBuffer()->markConstant(reallyConstant);
  }

  void setPrimary(void *ptr, size_t length);
  void setSpecial(void *ptr, size_t length);

  void expand(size_t newlength);

  /**
   * Set the workspace that backs this buffer. When workspace is set, the buffer
   * does not own its memory (workspace manages the lifecycle).
   * Pass nullptr to detach from workspace.
   */
  void setWorkspace(memory::Workspace* workspace);

  /**
   * Get the workspace backing this buffer (nullptr if heap-allocated).
   */
  memory::Workspace* getWorkspace() const { return _workspace; }

  /**
   * Check if this buffer is backed by a workspace (vs heap-allocated).
   */
  bool isWorkspaceBacked() const { return _workspace != nullptr; }

  /**
   * Set the multi-backend workspace for coherence tracking.
   * This enables cross-device coherence state optimization during sync operations.
   */
  void setMultiBackendWorkspace(memory::MultiBackendWorkspace* mbw) { _multiBackendWorkspace = mbw; }

  /**
   * Get the multi-backend workspace (nullptr if not set).
   */
  memory::MultiBackendWorkspace* getMultiBackendWorkspace() const { return _multiBackendWorkspace; }

  /**
   * Invalidate cached pointers. Must be called when workspace cycles
   * (scopeOut/scopeIn) to ensure stale cached pointers are not used.
   */
  void invalidateCachedPointers() {
    _cachedPrimaryPtr = nullptr;
    _cachedSpecialPtr = nullptr;
  }

  int deviceId() const;
  void setDeviceId(int deviceId);

  //updates whether the buffer is the owner of its associated buffers or not.
  void markOwner(bool owner);

  int useCount() const;

  static void registerSpecialUse(const std::vector<const InteropDataBuffer *> &writeList,
                                 const std::vector<const InteropDataBuffer *> &readList);
  static void prepareSpecialUse(const std::vector<const InteropDataBuffer *> &writeList,
                                const std::vector<const InteropDataBuffer *> &readList,
                                bool synchronizeWritables = false);

  static void registerPrimaryUse(const std::vector<const InteropDataBuffer *> &writeList,
                                 const std::vector<const InteropDataBuffer *> &readList);
  static void preparePrimaryUse(const std::vector<const InteropDataBuffer *> &writeList,
                                const std::vector<const InteropDataBuffer *> &readList,
                                bool synchronizeWritables = false);
};
}  // namespace sd

#endif  // LIBND4J_INTEROPDATABUFFER_H
