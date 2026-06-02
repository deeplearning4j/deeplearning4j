/* ******************************************************************************
 *
 * Copyright (c) 2024-2026 Contributors
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

#ifndef LIBND4J_DSP_BUFFER_POOL_H
#define LIBND4J_DSP_BUFFER_POOL_H

#include <system/common.h>
#include <array/DataBuffer.h>
#include <array/DataType.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

// ═══════════════════════════════════════════════════════════════════════════════
// DspBufferPool — per-device pool of reusable GPU DataBuffers
// ═══════════════════════════════════════════════════════════════════════════════
//
// Buffers are keyed by (numElements, dtype).  When a plan releases a buffer,
// it goes into the pool.  When another plan (or the same plan after passivation
// + reactivation) needs a buffer of the same shape, it acquires from the pool
// instead of calling cudaMalloc.
//
// Cross-plan sharing: Plan A (decode, active) releases colored buffers →
// Plan B (prefill, inactive/passivated) can acquire them.  For same-model
// plans with matching intermediate shapes, pool hit rate is very high.
//
// Thread safety: all public methods are mutex-guarded.  The pool is per-device
// so contention is limited to plans on the same GPU.
//
// Ownership: The pool owns DataBuffers while they are pooled (between release
// and acquire).  acquire() transfers ownership to the caller.  release()
// transfers ownership to the pool.

class SD_LIB_EXPORT DspBufferPool {
 public:
  /**
   * Get the pool singleton for a given device.
   * Lazily created on first access.
   */
  static DspBufferPool& forDevice(int deviceId);

  /**
   * Get the pool for the current device (uses AffinityManager).
   * No platform guards needed at call sites.
   */
  static DspBufferPool& forCurrentDevice();

  /**
   * Destroy all pool singletons.  Called during shutdown.
   */
  static void destroyAll();

  // ── Acquire / Release ──────────────────────────────────────────────────

  /**
   * Acquire a DataBuffer of the given size and dtype.
   * If the pool has a matching buffer, return it (zero-copy reuse).
   * Otherwise allocate fresh.
   *
   * @param numElements  Number of elements
   * @param dtype        Data type
   * @return Owned DataBuffer pointer.  Caller must release() or delete.
   */
  DataBuffer* acquire(LongType numElements, DataType dtype);

  /**
   * Release a DataBuffer back to the pool for reuse.
   * Buffer is NOT freed — it sits in the pool until acquired or trimmed.
   * Caller transfers ownership to the pool.
   *
   * @param buffer       DataBuffer to release (must not be nullptr)
   * @param numElements  Number of elements (for pool keying)
   * @param dtype        Data type (for pool keying)
   */
  void release(DataBuffer* buffer, LongType numElements, DataType dtype);

  // ── Memory pressure ────────────────────────────────────────────────────

  /**
   * Free pooled buffers until at least targetFreeBytes have been freed.
   * Called by the plan cache when memory is tight.
   *
   * @param targetFreeBytes  How many bytes to free (0 = free everything)
   * @return Actual bytes freed.
   */
  size_t trim(size_t targetFreeBytes);

  // ── Introspection ──────────────────────────────────────────────────────

  size_t pooledBytes() const;
  int pooledCount() const;
  size_t totalAcquired() const { return totalAcquired_.load(std::memory_order_relaxed); }
  size_t totalReused() const { return totalReused_.load(std::memory_order_relaxed); }
  int deviceId() const { return deviceId_; }

  struct PoolReport {
    int deviceId;
    size_t pooledBytes;
    int pooledCount;
    int numShapeClasses;
    size_t totalAcquired;
    size_t totalReused;
    float reuseRate;
  };

  PoolReport report() const;

 private:
  explicit DspBufferPool(int deviceId);
  ~DspBufferPool();

  // Non-copyable
  DspBufferPool(const DspBufferPool&) = delete;
  DspBufferPool& operator=(const DspBufferPool&) = delete;

  int deviceId_;

  // Key: hash of (numElements, dtype).  Value: list of available buffers.
  std::unordered_map<uint64_t, std::vector<DataBuffer*>> available_;

  // Track all buffers currently in the pool (for double-release detection)
  std::unordered_set<DataBuffer*> pooledSet_;

  mutable std::mutex mutex_;

  // Stats
  std::atomic<size_t> totalAcquired_{0};
  std::atomic<size_t> totalReused_{0};
  std::atomic<size_t> pooledBytes_{0};
  std::atomic<int> pooledCount_{0};

  /**
   * Compute pool key from (numElements, dtype).
   */
  static uint64_t keyFor(LongType numElements, DataType dtype);

  // Global pool storage
  static std::mutex globalMutex_;
  static std::unordered_map<int, DspBufferPool*> pools_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_BUFFER_POOL_H
