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

#include <graph/NativePlanCache.h>
#include <graph/DspBufferPool.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <memory/MemoryUtils.h>
#include <system/Environment.h>

#include <graph/DspDeviceDispatch.h>

namespace sd {
namespace graph {

// ---------------------------------------------------------------------------
// Static members
// ---------------------------------------------------------------------------

std::atomic<bool> NativePlanCache::shutdownInProgress_{false};

void NativePlanCache::setShutdownInProgress(bool inProgress) {
  shutdownInProgress_.store(inProgress, std::memory_order_release);
}

bool NativePlanCache::isShutdownInProgress() {
  return shutdownInProgress_.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// hashShapeInfoContents — content-based hash of placeholder shape-info buffers
// ---------------------------------------------------------------------------

uint64_t NativePlanCache::hashShapeInfoContents(sd::LongType** ptrs, sd::LongType count) {
  // FNV-1a 64-bit over the raw LongType words of each shape-info buffer.
  // Each buffer is rank*2+4 LongType elements (rank, dims, strides, extras).
  uint64_t h = 14695981039346656037ULL;
  for (sd::LongType i = 0; i < count; i++) {
    const sd::LongType* si = ptrs[i];
    if (si == nullptr) {
      // Mix a sentinel for null pointers.
      h ^= 0xDEAD;
      h *= 1099511628211ULL;
      continue;
    }
    sd::LongType rank = shape::rank(si);
    sd::LongType len = shape::shapeInfoLength(rank);
    for (sd::LongType j = 0; j < len; j++) {
      h ^= static_cast<uint64_t>(si[j]);
      h *= 1099511628211ULL;
    }
  }
  return h;
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

NativePlanCache::NativePlanCache() = default;

NativePlanCache::~NativePlanCache() {
  clear();
}

// ---------------------------------------------------------------------------
// size
// ---------------------------------------------------------------------------

size_t NativePlanCache::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return lru_.size();
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

void NativePlanCache::clear() {
  // During JVM shutdown, skip all CUDA teardown — the CUDA context may
  // already be destroyed.  The OS reclaims all memory on process exit.
  if (isShutdownInProgress()) {
    return;
  }

  // Collect all plans under the lock, then delete outside to avoid
  // running CUDA teardown (platformFreePlanResources) under mutex_.
  std::vector<NativeDynamicShapePlan*> toDelete;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    toDelete.reserve(lru_.size());
    for (auto& entry : lru_) {
      toDelete.push_back(entry.second);
    }
    lru_.clear();
    map_.clear();
    pinnedPlans_.clear();
  }

  for (auto* plan : toDelete) {
    delete plan;
  }
}

// ---------------------------------------------------------------------------
// getOrInsert
// ---------------------------------------------------------------------------

NativeDynamicShapePlan* NativePlanCache::getOrInsert(
    const Key& key,
    const std::function<NativeDynamicShapePlan*()>& factory) {

  NativeDynamicShapePlan* result = nullptr;
  std::vector<NativeDynamicShapePlan*> victims;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Cache hit: splice to front (MRU), pin, and return.
    auto it = map_.find(key);
    if (it != map_.end()) {
      lru_.splice(lru_.begin(), lru_, it->second);
      NativeDynamicShapePlan* plan = it->second->second;
      pinnedPlans_.insert(plan);
      // Reactivate passivated plans — execute path re-warms automatically.
      if (plan->isPassivated()) {
        plan->reactivate();
        DSP_DIAG(EXECUTE, "PLAN_CACHE HIT (reactivated) thread=0x%llx plan=%p shapeHash=0x%llx",
                 (unsigned long long)key.threadId, (void*)plan,
                 (unsigned long long)key.phShapeContentHash);
      } else {
        DSP_DIAG(EXECUTE, "PLAN_CACHE HIT thread=0x%llx plan=%p shapeHash=0x%llx",
                 (unsigned long long)key.threadId, (void*)plan,
                 (unsigned long long)key.phShapeContentHash);
      }
      result = plan;
    } else {
      // Cache miss: build a new plan for this thread.
      // Check if another thread already has a plan with the same structure
      // (same outputs + shapes + mode, different threadId).  Log lineage
      // so operators can trace which thread's plan was the "donor" structure.
      NativeDynamicShapePlan* donorPlan = nullptr;
      for (auto& entry : lru_) {
        const Key& existing = entry.first;
        if (existing.outputSetHash == key.outputSetHash
            && existing.phShapeContentHash == key.phShapeContentHash
            && existing.phCount == key.phCount
            && existing.graphExecutionMode == key.graphExecutionMode
            && existing.threadId != key.threadId) {
          donorPlan = entry.second;
          break;
        }
      }

      NativeDynamicShapePlan* plan = factory();
      if (!plan) {
        return nullptr;
      }

      if (donorPlan != nullptr) {
        DSP_DIAG(EXECUTE, "PLAN_CACHE THREAD_DUP thread=0x%llx newPlan=%p donorPlan=%p "
                 "structure re-deserialized, exec state fresh",
                 (unsigned long long)key.threadId, (void*)plan, (void*)donorPlan);
      } else {
        DSP_DIAG(EXECUTE, "PLAN_CACHE NEW_PLAN thread=0x%llx plan=%p shapeHash=0x%llx "
                 "(first plan for this structure)",
                 (unsigned long long)key.threadId, (void*)plan,
                 (unsigned long long)key.phShapeContentHash);
      }

      // Insert at MRU (front) and pin.
      lru_.emplace_front(key, plan);
      map_[key] = lru_.begin();
      pinnedPlans_.insert(plan);

      // Enforce count and memory budgets (skips pinned plans).
      // Victims are returned, NOT deleted under the lock.
      victims = evictIfOverBudgetLocked();

      result = plan;
    }
  }  // mutex_ released here

  // Delete evicted plans OUTSIDE the mutex.  Plan destructors call
  // platformFreePlanResources() which makes CUDA API calls — these must
  // not run under the cache lock to avoid blocking concurrent lookups
  // and to prevent CUDA deadlocks.
  for (auto* victim : victims) {
    delete victim;
  }

  return result;
}

// ---------------------------------------------------------------------------
// unpinPlan
// ---------------------------------------------------------------------------

void NativePlanCache::unpinPlan(NativeDynamicShapePlan* plan) {
  if (!plan) return;

  // During shutdown, skip entirely — no eviction, no CUDA teardown.
  if (isShutdownInProgress()) return;

  // IMPORTANT: unpinPlan does NOT trigger memory-budget eviction.
  // Running evictIfOverBudgetLocked() here caused a passivation death loop:
  //   1. Plan executes, gets unpinned
  //   2. Budget check sees plan's memory > 25% of freeMem → passivates it
  //   3. passivate() destroys ALL CUDA graphs and resets lifecycle to SLOT_BY_SLOT
  //   4. reactivate() only clears a flag — restores nothing
  //   5. Plan must re-warm (4+ executions to reach REPLAYING)
  //   6. Gets passivated again before reaching REPLAYING → never stabilizes
  // Memory-budget eviction only fires on getOrInsert() when new plans enter.
  std::lock_guard<std::mutex> lock(mutex_);
  pinnedPlans_.erase(plan);
}

// ---------------------------------------------------------------------------
// pinnedCount
// ---------------------------------------------------------------------------

size_t NativePlanCache::pinnedCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pinnedPlans_.size();
}

// ---------------------------------------------------------------------------
// evictIfOverBudgetLocked  (must be called with mutex_ held)
// ---------------------------------------------------------------------------

std::vector<NativeDynamicShapePlan*> NativePlanCache::evictIfOverBudgetLocked() {
  std::vector<NativeDynamicShapePlan*> victims;

  // During shutdown, skip eviction entirely.
  if (isShutdownInProgress()) {
    return victims;
  }

  auto& dsp = sd::Environment::getInstance().dsp();

  // Device-resident plans share the accelerator cache budget independent of vendor.
  const int maxPlans = dspHasDeviceMemory() ? dsp.planCacheMaxPlans() : dsp.planCacheMaxPlansCpu();
  const float fraction = dsp.planCacheBudgetFraction();

  // Helper lambda: find oldest unpinned, non-passivated plan (LRU end toward MRU front)
  auto findPassivationCandidate = [&]() -> LruList::iterator {
    for (auto it = std::prev(lru_.end()); ; ) {
      if (pinnedPlans_.count(it->second) == 0 && !it->second->isPassivated()) {
        return it;
      }
      if (it == lru_.begin()) break;
      --it;
    }
    return lru_.end();
  };

  // Helper lambda: find oldest unpinned plan (for full eviction, includes passivated)
  auto findEvictionCandidate = [&]() -> LruList::iterator {
    for (auto it = std::prev(lru_.end()); ; ) {
      if (pinnedPlans_.count(it->second) == 0) {
        return it;
      }
      if (it == lru_.begin()) break;
      --it;
    }
    return lru_.end();
  };

  // Helper: compute total cache footprint (only non-passivated plans count)
  auto computeCacheBytes = [&]() -> size_t {
    size_t total = 0;
    for (auto& entry : lru_) {
      if (entry.second->isPassivated()) continue;  // holds zero GPU memory
      size_t planBytes = entry.second->estimatedOwnedBytes();
      total += (planBytes > 0) ? planBytes : kBytesPerPlanEstimate;
    }
    return total;
  };

  // Helper: query device memory for budget computation.
  // GPU: uses total device memory (not free — using free memory is self-defeating
  // because the plan's own allocations reduce freeMem, shrinking the budget).
  // CPU: uses free system RAM (self-defeating loop doesn't apply — CPU plans
  // don't hold GPU-like exclusive memory and free RAM is usually abundant).
  auto queryDeviceMemForBudget = []() -> size_t {
    // GPU: use total device memory (not free — self-defeating).
    // CPU: use free system RAM.
    size_t gpuMem = dspGetDeviceTotalMemory();
    return (gpuMem > 0) ? gpuMem : sd::memory::MemoryUtils::getSystemFreeMemoryBytes();
  };

  // ═══════════════════════════════════════════════════════════════════════
  // Round 1: Hard count cap — full eviction for plans over max count
  // ═══════════════════════════════════════════════════════════════════════
  while (static_cast<int>(lru_.size()) > maxPlans) {
    auto victim = findEvictionCandidate();
    if (victim == lru_.end()) break;  // all pinned
    DSP_DIAG(MEMORY, "PLAN_CACHE evict LRU (count cap %d): outputSetHash=%llu phCount=%lld contentHash=0x%016llx",
             maxPlans,
             (unsigned long long)victim->first.outputSetHash,
             (long long)victim->first.phCount,
             (unsigned long long)victim->first.phShapeContentHash);
    NativeDynamicShapePlan* plan = victim->second;
    map_.erase(victim->first);
    lru_.erase(victim);
    victims.push_back(plan);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Memory-based eviction (rounds 2-4) — only when budget fraction is set
  // ═══════════════════════════════════════════════════════════════════════
  if (fraction > 0.0f && !lru_.empty()) {
    size_t totalDeviceMem = queryDeviceMemForBudget();
    if (totalDeviceMem > 0) {
      const size_t budgetBytes = static_cast<size_t>(fraction * static_cast<float>(totalDeviceMem));
      size_t totalCacheBytes = computeCacheBytes();

      // ─────────────────────────────────────────────────────────────────
      // Round 2: Passivation — release GPU intermediates, plan stays in cache
      // ─────────────────────────────────────────────────────────────────
      while (totalCacheBytes > budgetBytes) {
        auto candidate = findPassivationCandidate();
        if (candidate == lru_.end()) break;  // all pinned or already passivated

        NativeDynamicShapePlan* plan = candidate->second;
        size_t planBytes = plan->estimatedOwnedBytes();
        size_t planCost = (planBytes > 0) ? planBytes : kBytesPerPlanEstimate;

        DSP_DIAG(MEMORY, "PLAN_CACHE passivate LRU (memory budget %.1f%% of %zuMB total, "
                 "cache=%zuMB): plan=%p planBytes=%zuMB",
                 fraction * 100.0f,
                 totalDeviceMem / (1024 * 1024),
                 totalCacheBytes / (1024 * 1024),
                 (void*)plan,
                 planCost / (1024 * 1024));
        plan->passivate();
        totalCacheBytes -= planCost;
      }

      // ─────────────────────────────────────────────────────────────────
      // Round 3: Pool trim — free pooled buffers from DspBufferPool
      // ─────────────────────────────────────────────────────────────────
      if (totalCacheBytes > budgetBytes) {
        size_t overshoot = totalCacheBytes - budgetBytes;
        auto& pool = DspBufferPool::forCurrentDevice();
        size_t poolFreed = pool.trim(overshoot);
        DSP_DIAG(MEMORY, "PLAN_CACHE pool trim: freed %zuMB from buffer pool (target=%zuMB)",
                 poolFreed / (1024 * 1024), overshoot / (1024 * 1024));
        // Re-check: passivated plans now report ~0 bytes, recalculate
        totalCacheBytes = computeCacheBytes();
      }

      // ─────────────────────────────────────────────────────────────────
      // Round 4: Full eviction — delete LRU plans entirely
      // ─────────────────────────────────────────────────────────────────
      while (totalCacheBytes > budgetBytes && !lru_.empty()) {
        auto victim = findEvictionCandidate();
        if (victim == lru_.end()) break;  // all pinned

        size_t victimBytes = victim->second->estimatedOwnedBytes();
        size_t victimCost = (victimBytes > 0) ? victimBytes : kBytesPerPlanEstimate;

        DSP_DIAG(MEMORY, "PLAN_CACHE evict LRU (memory budget %.1f%% of %zuMB total, cache=%zuMB): "
                 "outputSetHash=%llu phCount=%lld contentHash=0x%016llx planBytes=%zuMB",
                 fraction * 100.0f,
                 totalDeviceMem / (1024 * 1024),
                 totalCacheBytes / (1024 * 1024),
                 (unsigned long long)victim->first.outputSetHash,
                 (long long)victim->first.phCount,
                 (unsigned long long)victim->first.phShapeContentHash,
                 victimCost / (1024 * 1024));
        totalCacheBytes -= victimCost;
        NativeDynamicShapePlan* plan = victim->second;
        map_.erase(victim->first);
        lru_.erase(victim);
        victims.push_back(plan);
      }
    }
  }

  return victims;
}

}  // namespace graph
}  // namespace sd
