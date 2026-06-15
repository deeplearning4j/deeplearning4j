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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef DEV_TESTS_DATABUFFER_H
#define DEV_TESTS_DATABUFFER_H

#include <array/DataType.h>
#include <execution/LaunchContext.h>
#include <memory/Workspace.h>
#include <system/common.h>
#include <system/op_boilerplate.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
namespace sd {

#ifndef __JAVACPP_HACK__
/**
 * Thread-local flag indicating graph execution/capture is in progress.
 * When true, syncToPrimary() skips D2H transfers -- data must stay
 * on the compute device during graph execution. Applies to all graph
 * backends: CUDA Graphs (capture/replay), oneDNN Graph, and ACL
 * Dynamic Fusion. Set by NativeDynamicShapePlan around graph segment
 * execution, cleared when execution completes or is aborted.
 */
extern SD_TLS_EXPORT thread_local bool tl_graphExecutionActive;

#ifdef SD_CUDA
/**
 * Set during DSP composite replay gap-slot execution. Suppresses post-op
 * cudaStreamSynchronize calls in PointersManager, cuDNN ops, etc.
 * Unlike tl_graphExecutionActive, this does NOT affect allocation or
 * memory freeing behavior — it only signals that all work is on a single
 * unified stream (tl_dspGapStream) where FIFO ordering makes per-op
 * syncs redundant.
 */
extern SD_TLS_EXPORT thread_local bool tl_dspReplayActive;

/**
 * Captured CUDA stream for the current graph capture session.
 * Some capture-safe paths must enqueue work on the exact captured stream;
 * using a different stream can invalidate capture.
 */
extern SD_TLS_EXPORT thread_local cudaStream_t tl_graphCaptureStream;
#endif

/**
 * Thread-local accumulator for pinned host buffers allocated during CUDA graph capture.
 * PointersManager::replicatePointer copies host data to persistent pinned memory
 * during capture so graph replay reads from valid addresses.
 * After capture, these are transferred to CudaGraphHandle for lifetime management.
 */
extern SD_TLS_EXPORT thread_local std::vector<void*> tl_capturedHostPtrs;
/**
 * Thread-local cache for PointersManager H2D copies during CUDA graph capture.
 * Maps {content_hash ^ size} -> device pointer. When the same data is uploaded
 * multiple times during capture (e.g., dimension arrays [0,1] used by many ops),
 * the cached device pointer is returned without creating a redundant memcpy node.
 * Cleared at the start of each capture.
 */
extern SD_TLS_EXPORT thread_local std::unordered_map<uint64_t, void*> tl_captureReplicateCache;

/**
 * Capture workspace: pre-allocated GPU buffer used during CUDA graph capture
 * to eliminate cudaMallocAsync/cudaFreeAsync nodes from the captured graph.
 * CudaMemoryPool::allocate uses bump allocation from this workspace instead of
 * cudaMallocAsync when tl_graphExecutionActive && tl_captureWorkspace != nullptr.
 * CudaMemoryPool::free becomes a no-op for addresses within this workspace.
 * The workspace buffer persists for graph lifetime (stored on GraphSegment).
 */
extern SD_TLS_EXPORT thread_local void* tl_captureWorkspace;
extern SD_TLS_EXPORT thread_local size_t tl_captureWorkspaceSize;
extern SD_TLS_EXPORT thread_local size_t tl_captureWorkspaceOffset;

/**
 * Capture host workspace: pre-allocated PINNED HOST buffer used during CUDA
 * graph capture as H2D memcpy source. DataBuffer::syncToSpecial and
 * PointersManager bump-allocate from this workspace instead of using
 * _primaryBuffer directly. Temporary host arrays (axis/dimension params for
 * gap ops) get freed after the op completes, but the graph's H2D node bakes
 * the source address — reading freed memory on replay causes SIGSEGV.
 * The pinned workspace persists for the graph's lifetime via tl_capturedHostPtrs.
 */
extern SD_TLS_EXPORT thread_local void* tl_captureHostWorkspace;
extern SD_TLS_EXPORT thread_local size_t tl_captureHostWorkspaceSize;
extern SD_TLS_EXPORT thread_local size_t tl_captureHostWorkspaceOffset;

/**
 * cuBLAS workspace buffer and size for graph capture.
 * Set by NativeDynamicShapePlan::setCublasWorkspaceForCapture().
 * Read by MmulHelper after cublasSetStream resets workspace.
 * (cublasSetStream resets user-provided workspace per cuBLAS docs.)
 */
extern SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr;
extern SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize;


/**
 * Per-step GPU allocation/free tracking for leak diagnosis.
 * Reset at segment entry, logged at segment exit via DSP_DIAG.
 */
extern SD_TLS_EXPORT thread_local long long tl_dspAllocBytes;
extern SD_TLS_EXPORT thread_local long long tl_dspFreeBytes;
extern SD_TLS_EXPORT thread_local int tl_dspAllocCount;
extern SD_TLS_EXPORT thread_local int tl_dspFreeCount;
extern SD_TLS_EXPORT thread_local int tl_dspFreeSkipCount;

#ifdef SD_CUDA
/**
 * DSP execution stream override for syncToSpecial().
 * When set, syncToSpecial uses this stream instead of stream 0 and skips
 * per-call cudaStreamSynchronize (caller guarantees ordering via same-stream
 * graph launch). Set/unset by DSP replay path around ext input sync loops.
 */
extern SD_TLS_EXPORT thread_local cudaStream_t tl_dspExecutionStream;

/**
 * Per-island slot range filter for composite CUDA graph capture.
 * When tl_islandSlotMin <= tl_islandSlotMax, TritonGraphBackend::executeSegment
 * skips sub-kernels whose slot range falls entirely outside [tl_islandSlotMin,
 * tl_islandSlotMax]. This allows capturing a single Triton island from a
 * composite (mixed Triton/gap) segment without capturing other islands.
 * When tl_islandSlotMin > tl_islandSlotMax, no filtering is applied.
 * Set by the per-island capture loop in NativeDynamicShapePlan_gpubackend.cpp,
 * cleared after each island's capture completes.
 */
extern SD_TLS_EXPORT thread_local int tl_islandSlotMin;
extern SD_TLS_EXPORT thread_local int tl_islandSlotMax;
#endif
#endif  // __JAVACPP_HACK__

class SD_LIB_EXPORT DataBuffer {
 private:
  // Magic number for validity checking (pattern from DirectShapeTrie validation)
  // Set in constructor, cleared in destructor, checked before use
  // Helps detect use-after-free and corrupted pointers
  static constexpr uint32_t MAGIC_NUMBER = 0xDA7ABF01;  // "DA7ABF01" (DataBuffer v01)

  // Padding appended to non-workspace host allocations to absorb minor buffer overruns.
  // The padding region is filled with canary values (0xDEADBEEFCAFEBABE) so that
  // deletePrimary() can detect overruns before calling free().
  static constexpr sd::LongType HOST_ALLOC_PADDING = 65536;
  uint32_t _magicNumber = MAGIC_NUMBER;

  void *_primaryBuffer = nullptr;
  void *_specialBuffer = nullptr;
  std::atomic<int> _specialDeviceId{-1};  // Device ID where _specialBuffer was allocated (for multi-GPU)
  LongType _lenInBytes = 0;
  // Track actual allocated sizes independently to prevent overrun when
  // setPrimaryBuffer/setSpecialBuffer are called with different sizes.
  LongType _primaryAllocBytes = 0;
  LongType _specialAllocBytes = 0;
  memory::Workspace *_workspace = nullptr;

  std::atomic<int> _deviceId;
  std::mutex _deleteMutex;
#ifndef __JAVACPP_HACK__
#if defined(SD_CUDA)
  mutable std::atomic<LongType> _counter;
  mutable std::atomic<LongType> _writePrimary;
  mutable std::atomic<LongType> _writeSpecial;
  mutable std::atomic<LongType> _readPrimary;
  mutable std::atomic<LongType> _readSpecial;

  // CUDA event to track the last write to special (device) buffer.
  // This enables stream-ordered consumers to wait on async D2D/H2D writes without
  // draining the device. The event is created lazily and stores the cudaEvent_t
  // value directly (no heap allocation in the hot path).
  mutable void* _writeEvent = nullptr;  // cudaEvent_t handle, void* to avoid cuda_runtime.h in header
  mutable std::atomic<int> _writeEventDeviceId{-1};
  mutable std::atomic<bool> _writeEventRecorded{false};
#endif

#if defined(SD_GCC_FUNCTRACE)
  StackTrace *allocationStackTracePrimary = nullptr;
  StackTrace *allocationStackTraceSpecial = nullptr;
  StackTrace *creationStackTrace = nullptr;

#endif


#endif

  bool closed = false;

  /**
   * Frozen plan reference count. When > 0, this DataBuffer's _specialBuffer is
   * registered in one or more frozen NativeDynamicShapePlan contexts. Migrating
   * (i.e., freeing and re-allocating _specialBuffer on a different device) would
   * invalidate the baked-in GPU addresses that frozen slot contexts or CUDA graphs
   * rely on for replay. DataBuffer::migrate() skips migration when this count > 0.
   *
   * Lifecycle:
   *   - Incremented by NativeDynamicShapePlan when a buffer is registered as an
   *     external input or retained weight for frozen execution.
   *   - Decremented by releaseGpuIntermediates() when the frozen plan is torn down.
   *   - Atomic because multiple plans could theoretically share the same buffer.
   */
  std::atomic<int> _frozenRefCount{0};

  // Helper template function for printing host buffer content (implementation in .cpp)
  template <typename T>
  void printHostBufferContent(void* buffer, sd::LongType offset, sd::LongType length);

  /**
   * Frozen-phase mutation guard. Throws an exception with a detailed message
   * if this DataBuffer has one or more frozen references registered
   * (_frozenRefCount > 0). Call from the top of any method that would mutate
   * the identity of the backing storage (reallocate, free, setPrimaryBuffer,
   * setSpecialBuffer, replaceSpecialBuffer, expand, migrate, close, etc.).
   *
   * Methods that only copy CONTENT (syncToPrimary/syncToSpecial/writePrimary/
   * writeSpecial/readPrimary/readSpecial) do NOT need this guard — they don't
   * change the underlying pointer, so frozen contexts that baked the address
   * remain valid.
   *
   * @param op human-readable name of the calling mutator method
   */
  void throwIfFrozen(const char* op) const;

  void setCountersToZero();
  void copyCounters(const DataBuffer &other);
  void deleteSpecial();
  void deletePrimary();
  void setAllocFlags(const bool isOwnerPrimary, const bool isOwnerSpecial = false);
  void allocateBuffers(const bool allocBoth = false);

  void setSpecial(void *special, const bool isOwnerSpecial);

  void copyBufferFromHost(const void *hostBuffer, size_t sizeToCopyinBytes = 0, const LongType offsetThis = 0,
                          const LongType offsetHostBuffer = 0);

 public:

  void deleteBuffers();

  /**
   * Free GPU (special) buffer only, abandon host (primary) buffer.
   * Used by dbFreeBuffersOnly to avoid SIGABRT from host heap corruption.
   */
  void freeGpuOnly();

  /**
   * Free GPU (special) buffer on the specified CUDA stream.
   * Used by DSP mid-execution flushing to free on the execution stream
   * instead of stream 0, so the pool can reuse memory on the same stream.
   * @param stream CUDA stream for cudaFreeAsync (nullptr = default stream)
   */
  void freeGpuOnStream(void* stream);

  bool _isOwnerPrimary;
  bool _isOwnerSpecial;
  bool isConstant = false;
  DataType _dataType;

  DataBuffer(void *primary, void *special, const size_t lenInBytes, const DataType dataType,
             const bool isOwnerPrimary = false, const bool isOwnerSpecial = false,
             memory::Workspace *workspace = nullptr);

  DataBuffer(void *primary, const size_t lenInBytes, const DataType dataType, const bool isOwnerPrimary = false,
             memory::Workspace *workspace = nullptr);

  DataBuffer(const void *hostBuffer,  // copies data from hostBuffer to own memory buffer
             const DataType dataType, const size_t lenInBytes, memory::Workspace *workspace = nullptr);

  DataBuffer(const sd::LongType lenInBytes, const DataType dataType, memory::Workspace *workspace = nullptr,
             const bool allocBoth = false);

  DataBuffer(const DataBuffer &other);
  DataBuffer(DataBuffer &&other);
  explicit DataBuffer();
  ~DataBuffer();

  DataBuffer &operator=(const DataBuffer &other);
  DataBuffer &operator=(DataBuffer &&other) noexcept;

  DataType getDataType();
  void setDataType(DataType dataType);
  size_t getLenInBytes() const;

  size_t getNumElements();

  template <typename T>
  void *primaryAtOffset(const LongType offset);
  template <typename T>
  void *specialAtOffset(const LongType offset);

  void *primary();
  void *special();
  void printAllocationTrace();

  /**
   * Validate that this DataBuffer object is in a sane state.
   * Following DirectShapeTrie validation pattern: check magic number, closed flag, etc.
   * Throws exception with detailed message if validation fails.
   * Call this before accessing any member in methods that might be called
   * on dangling/corrupted pointers (like special(), primary(), etc.)
   */
  void validateIntegrity() const;

  /**
   * Check if this DataBuffer has been destroyed (destructor was called).
   * After destruction, the magic number is set to 0xDEADBEEF.
   * This is useful for detecting use-after-free and preventing double-free.
   * @return true if the buffer has been destroyed, false if it's still valid
   */
  bool isDestroyed() const { return _magicNumber == 0xDEADBEEF; }

  /**
   * Check if this DataBuffer is valid (not destroyed and has correct magic number).
   * @return true if the buffer is valid, false otherwise
   */
  bool isValid() const { return _magicNumber == MAGIC_NUMBER && !closed; }

#if defined(SD_CUDA)
  void* writeEvent() const { return _writeEvent; }
  bool writeEventRecorded() const { return _writeEventRecorded.load(std::memory_order_acquire); }
  void waitForSpecialWriteEvent(void* stream) const;
  void recordSpecialWriteEvent(void* stream) const;
  void clearSpecialWriteEvent() const;
#else
  void* writeEvent() const { return nullptr; }
  bool writeEventRecorded() const { return false; }
  void waitForSpecialWriteEvent(void* stream) const {}
  void recordSpecialWriteEvent(void* stream) const {}
  void clearSpecialWriteEvent() const {}
#endif

  void allocatePrimary();
  void allocateSpecial();

  void writePrimary() const;
  void writeSpecial() const;
  void readPrimary() const;
  void readSpecial() const;
  bool isPrimaryActual() const;
  bool isSpecialActual() const;

  void expand(const uint64_t size);

  int deviceId() const;
  void setDeviceId(int deviceId);
  void migrate();

  template <typename T>
  SD_INLINE T *primaryAsT();
  template <typename T>
  SD_INLINE T *specialAsT();

  void markConstant(bool reallyConstant);

  /**
   * Increment the frozen plan reference count. Call when this buffer is
   * registered in a frozen NativeDynamicShapePlan as an external input
   * or retained weight. While the count is > 0, migrate() is blocked to
   * prevent invalidating baked-in GPU addresses used by frozen replay.
   */
  void addFrozenRef() { _frozenRefCount.fetch_add(1, std::memory_order_relaxed); }

  /**
   * Decrement the frozen plan reference count. Call during
   * releaseGpuIntermediates() when the frozen plan is torn down.
   */
  void removeFrozenRef() {
    auto prev = _frozenRefCount.fetch_sub(1, std::memory_order_relaxed);
    if (prev <= 0) {
      _frozenRefCount.store(0, std::memory_order_relaxed);  // Clamp to 0
    }
  }

  /**
   * Check whether this buffer is registered in any frozen plan.
   * @return true if frozen ref count > 0
   */
  bool isFrozenPlanRegistered() const { return _frozenRefCount.load(std::memory_order_relaxed) > 0; }

  void syncToPrimary(const LaunchContext *context, const bool forceSync = false);
  void syncToSpecial(const bool forceSync = false);

  void setToZeroBuffers(const bool both = false);

  void copyBufferFrom(const DataBuffer &other, size_t sizeToCopyinBytes = 0, const LongType offsetThis = 0,
                      const LongType offsetOther = 0);


  void setPrimaryBuffer(void *buffer, size_t length);
  void setSpecialBuffer(void *buffer, size_t length);

  /**
   * Replace the special (device) buffer pointer WITHOUT freeing the old pointer.
   * The caller is responsible for freeing the old pointer separately.
   * Used by weight migration: the old pool-based pointer is freed via cudaFreeAsync,
   * then this method sets the new direct-allocated pointer.
   * On CPU this is a no-op since there is no device buffer.
   * @param newPtr  New special buffer pointer (caller must manage ownership)
   * @param isOwner Whether this DataBuffer owns (should free) the new pointer
   */
  void replaceSpecialBuffer(void* newPtr, bool isOwner);


  void  showBufferLimited();
  //for Debug purposes
  void showCounters(const char* msg1, const char* msg2);
  // Reset host/device sync counters when reusing buffers.
  void resetCounters();

  /**
   * This method deletes buffers, if we're owners
   */
  void close();
  bool isClosed() { return closed; }
  void printPrimaryAllocationStackTraces();
  void printSpecialAllocationTraces();
  DataBuffer  dup();

  /**
   * Helper method to format creation stack trace as string for error messages.
   * Returns formatted stack trace if SD_GCC_FUNCTRACE is enabled, empty string otherwise.
   */
  std::string getCreationTraceAsString() const;
  void printHostDevice(long offset);
  static void memcpy(DataBuffer *dst, DataBuffer *src, sd::LongType startingOffset, sd::LongType dstOffset, sd::LongType n = 0);
  /**
   * Print detailed buffer information including host and device content if available
   * @param msg - Optional message to display
   * @param offset - Starting offset for printing buffer contents
   * @param limit - Maximum number of elements to print
   */
#ifndef __JAVACPP_HACK__
  void printBufferDebug(const char* msg = nullptr, sd::LongType offset = 0, sd::LongType limit = 10);
#endif

  // Padded operator new/delete to protect adjacent glibc chunks from
  // overruns on nearby allocations. DataBuffer objects are ~200 bytes on
  // the heap with zero padding — any adjacent overrun corrupts the next
  // chunk metadata → SIGABRT on free(). Adding 4KB padding keeps the
  // next chunk's header safely out of reach.
  static void* operator new(size_t size) {
    return std::malloc(size + 4096);
  }
#ifndef __JAVACPP_HACK__
  static void* operator new(size_t size, const std::nothrow_t&) noexcept {
    return std::malloc(size + 4096);
  }
#endif
  static void operator delete(void* ptr) noexcept {
    std::free(ptr);
  }
#ifndef __JAVACPP_HACK__
  static void operator delete(void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
  }
#endif
};
///// IMPLEMENTATION OF INLINE METHODS /////

////////////////////////////////////////////////////////////////////////
template <typename T>
T *DataBuffer::primaryAsT() {
  return reinterpret_cast<T *>(_primaryBuffer);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T *DataBuffer::specialAsT() {
  return reinterpret_cast<T *>(_specialBuffer);
}

}  // namespace sd

#endif  // DEV_TESTS_DATABUFFER_H
