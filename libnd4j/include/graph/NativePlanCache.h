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

#ifndef LIBND4J_NATIVE_PLAN_CACHE_H
#define LIBND4J_NATIVE_PLAN_CACHE_H

#include <system/common.h>
#include <graph/NativeDynamicShapePlan.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sd {
namespace graph {

/**
 * LRU cache keyed by (outputSetHash, content-hash of ALL external input shape-info buffers).
 * The key hashes the actual shape-info CONTENTS (rank, dims, strides, dtype) rather
 * than pointer addresses, so arrays with the same shape but different allocations
 * produce the same key. All externals (placeholders, variables, constants) are included
 * so that any shape change (KV cache growth, mask resize) dispatches to a different plan.
 *
 * Budget policy (both enforced on every insert):
 *   1. Hard cap:    lru_.size() > DspConfig::planCacheMaxPlans()  → evict LRU until satisfied.
 *   2. Memory soft: estimated cache footprint > planCacheBudgetFraction * freeDeviceMem
 *      → evict LRU until under budget.  "Estimated footprint" is currently approximated
 *      as (entry_count * kBytesPerPlanEstimate); this is a conservative flat estimate
 *      until NativeDynamicShapePlan exposes a memoryUsage() API.
 *
 * Ownership: the cache owns every NativeDynamicShapePlan* it stores and deletes them on
 * eviction or destruction.
 *
 * Thread-safety: all public methods are guarded by an internal mutex.
 */
class SD_LIB_EXPORT NativePlanCache {
 public:
  // -------------------------------------------------------------------------
  // Key type — content-based (not pointer-based)
  // -------------------------------------------------------------------------
  struct Key {
    uint64_t outputSetHash;
    // Content hash of ALL external input shape-info buffers (placeholders + variables).
    // Built by hashShapeInfoContents() from the actual rank/dims/strides/dtype data.
    // Any external shape change (KV cache growth, mask resize) produces a different key.
    uint64_t phShapeContentHash;
    // Number of external inputs (for equality disambiguation).
    sd::LongType phCount;
    // GraphExecutionMode — each mode gets its own plan (one flow, no reclassification).
    int graphExecutionMode;
    // Thread isolation: each thread gets its own plan instance so that mutable
    // execution state (outputSlots_, staging buffers, segment lifecycle) is never
    // shared across concurrent sd.output() calls.  The plan structure (slots,
    // segment definitions) is re-deserialized from the cached bytes — this is
    // cheap (~µs) and avoids adding synchronization to the hot execution path.
    uint64_t threadId = 0;
    // FNV-1a over the FULL serialized plan bytes — exact structural identity.
    // Without it, two DIFFERENT graphs that happen to share the same sorted
    // output-name set, placeholder shape contents, count, mode, and thread
    // collide and serve each other's plans (observed cross-test: a 3-external
    // 'b'-input plan served to longViewChain → "missing external input 'b'").
    // Hashed once per dispatchNativePlan call (plan acquire / shape change),
    // never on the per-exec hot path.
    uint64_t planContentHash = 0;

    bool operator==(const Key& o) const {
      return outputSetHash == o.outputSetHash
          && phShapeContentHash == o.phShapeContentHash
          && phCount == o.phCount
          && graphExecutionMode == o.graphExecutionMode
          && threadId == o.threadId
          && planContentHash == o.planContentHash;
    }
  };

  struct KeyHasher {
    size_t operator()(const Key& k) const noexcept {
      size_t h = std::hash<uint64_t>{}(k.outputSetHash);
      h ^= std::hash<uint64_t>{}(k.phShapeContentHash) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h ^= std::hash<sd::LongType>{}(k.phCount) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h ^= std::hash<int>{}(k.graphExecutionMode) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h ^= std::hash<uint64_t>{}(k.threadId) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h ^= std::hash<uint64_t>{}(k.planContentHash) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };

  /**
   * Hash the contents of an array of shape-info pointers. Reads the actual
   * rank/dims/strides/dtype from each buffer to produce a stable content hash
   * that is independent of pointer addresses.
   */
  static uint64_t hashShapeInfoContents(sd::LongType** ptrs, sd::LongType count);

  // -------------------------------------------------------------------------
  // Shutdown guard
  // -------------------------------------------------------------------------

  /**
   * Mark the plan cache subsystem as shutting down.  When set, unpinPlan()
   * and evictIfOverBudgetLocked() skip plan deletion entirely — the OS
   * reclaims all memory on process exit.  This prevents SIGSEGV from CUDA
   * API calls (cudaStreamDestroy, cudaFree, cudaGraphExecDestroy) racing
   * with JVM shutdown tearing down the CUDA context.
   *
   * Called from the JVM shutdown hook via setPlanCacheShutdownInProgress().
   */
  static void setShutdownInProgress(bool inProgress);
  static bool isShutdownInProgress();

  // -------------------------------------------------------------------------
  // Construction / destruction
  // -------------------------------------------------------------------------
  NativePlanCache();
  ~NativePlanCache();

  // Non-copyable, non-movable: owns raw pointers and carries a mutex.
  NativePlanCache(const NativePlanCache&) = delete;
  NativePlanCache& operator=(const NativePlanCache&) = delete;
  NativePlanCache(NativePlanCache&&) = delete;
  NativePlanCache& operator=(NativePlanCache&&) = delete;

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Lookup `key`.  If found, promote to MRU position and return the stored plan.
   * If not found, call `factory()` to create a new plan, insert it at MRU, run
   * budget enforcement (evictIfOverBudgetLocked), then return the new plan.
   *
   * The returned plan is automatically pinned (protected from eviction).
   * The caller MUST call unpinPlan() on the previously-held plan handle before
   * or after obtaining a new one, so it becomes eligible for eviction.
   *
   * Returns the plan pointer (owned by the cache).  Returns nullptr only if
   * `factory()` itself returns nullptr.
   */
  NativeDynamicShapePlan* getOrInsert(
      const Key& key,
      const std::function<NativeDynamicShapePlan*()>& factory);

  /**
   * Unpin a plan that was previously returned by getOrInsert().
   * Once unpinned, the plan becomes eligible for LRU eviction.
   * Safe to call with nullptr or a plan not in the cache (no-op).
   * Triggers eviction check after unpinning.
   */
  void unpinPlan(NativeDynamicShapePlan* plan);

  /**
   * Remove all entries, deleting every owned plan. Clears all pins.
   */
  void clear();

  /** Current number of cached entries. */
  size_t size() const;

  /** Current number of pinned (eviction-protected) plans. */
  size_t pinnedCount() const;

 private:
  // LRU list: front = most-recently-used, back = least-recently-used.
  using Entry    = std::pair<Key, NativeDynamicShapePlan*>;
  using LruList  = std::list<Entry>;
  using ListIter = LruList::iterator;

  mutable std::mutex mutex_;
  LruList lru_;
  std::unordered_map<Key, ListIter, KeyHasher> map_;

  // Plans currently in use by a Java executor. Eviction skips these.
  std::unordered_set<NativeDynamicShapePlan*> pinnedPlans_;

  // Process-global shutdown flag.  Set from JVM shutdown hook to prevent
  // CUDA API calls during context teardown.
  static std::atomic<bool> shutdownInProgress_;

  /**
   * Collect LRU entries that exceed the hard-count cap and memory-fraction
   * soft cap.  Skips pinned plans.  Returns victims WITHOUT deleting them;
   * callers must delete outside the mutex to avoid running CUDA teardown
   * (platformFreePlanResources) under the cache lock.
   *
   * Must be called with mutex_ already held.
   */
  std::vector<NativeDynamicShapePlan*> evictIfOverBudgetLocked();

  // Rough per-plan memory estimate used for the soft memory budget.
  // NativeDynamicShapePlan does not expose a memoryUsage() API; this flat constant
  // is used as a conservative per-plan upper bound until accurate accounting is added.
  static constexpr size_t kBytesPerPlanEstimate = 64ULL * 1024 * 1024;  // 64 MiB
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_NATIVE_PLAN_CACHE_H
