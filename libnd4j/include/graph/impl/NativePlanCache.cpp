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
std::mutex NativePlanCache::shutdownMutex_;

void NativePlanCache::setShutdownInProgress(bool inProgress) {
  std::lock_guard<std::mutex> lock(shutdownMutex_);
  shutdownInProgress_.store(inProgress, std::memory_order_release);
}

bool NativePlanCache::isShutdownInProgress() {
  return shutdownInProgress_.load(std::memory_order_acquire);
}

static void logPlanCacheMemoryState(
    const char* event, const NativePlanCache::Key* key,
    NativeDynamicShapePlan* plan, size_t entries, size_t pinned) {
  if (!DSP_DIAG_ENABLED(MEMORY)) return;

  auto& dsp = sd::Environment::getInstance().dsp();
  const size_t totalBytes = dspGetDeviceTotalMemory();
  DSP_DIAG(
      MEMORY,
      "PLAN_CACHE_STATE event=%s entries=%zu pinned=%zu maxPlans=%d "
      "budgetFraction=%.4f gpuTotal=%zuMB plan=%p "
      "thread=0x%llx "
      "shapeHash=0x%llx contentHash=0x%llx",
      event, entries, pinned,
      dspHasDeviceMemory() ? dsp.planCacheMaxPlans() : dsp.planCacheMaxPlansCpu(),
      dsp.planCacheBudgetFraction(),
      totalBytes / (1024 * 1024),
      (void*)plan,
      key != nullptr ? (unsigned long long)key->threadId : 0ULL,
      key != nullptr ? (unsigned long long)key->phShapeContentHash : 0ULL,
      key != nullptr ? (unsigned long long)key->planContentHash : 0ULL);
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
  // Serialize shutdown transitions with CUDA teardown. During JVM shutdown,
  // skip all CUDA teardown — the OS reclaims the remaining memory on exit.
  std::lock_guard<std::mutex> shutdownLock(shutdownMutex_);
  if (isShutdownInProgress()) return;

  // Collect only unleased plans. A live executor may still hold a raw plan
  // handle, so clear must not delete a leased entry; it remains in the cache
  // until the final borrower releases it.
  std::vector<NativeDynamicShapePlan*> toDelete;
  size_t leasedRemaining = 0;
  size_t entriesRemaining = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = lru_.begin(); it != lru_.end();) {
      NativeDynamicShapePlan* plan = it->second;
      if (pinCounts_.count(plan) != 0) {
        ++it;
        continue;
      }
      toDelete.push_back(plan);
      map_.erase(it->first);
      it = lru_.erase(it);
    }
    leasedRemaining = pinCounts_.size();
    entriesRemaining = lru_.size();
    clearPending_ = leasedRemaining != 0;
  }

  DSP_DIAG(MEMORY, "PLAN_CACHE_CLEAR: deleting=%zu leasedRemaining=%zu entriesRemaining=%zu",
           toDelete.size(), leasedRemaining, entriesRemaining);

  // Teardown is deliberately two-phase. Cached shape plans can share the same
  // external weight DataBuffers and each frozen plan pins those device pointers.
  // If plans are deleted one at a time, the first destructor still sees the
  // remaining plans' pins, while the final destructor drops the last pins only
  // after its platform migration opportunity has already passed. Persistent
  // weights then remain allocated from the async pool and fragment every later
  // generation by one model-sized allocation set.
  //
  // Release every plan first so the final release observes no other frozen
  // owners and can migrate the shared persistent weights out of the trim-managed
  // pool. Only after all release paths have run is it safe to destroy the plans.
  for (auto* plan : toDelete) {
    if (plan != nullptr) {
      plan->releaseGpuIntermediates();
    }
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
    const std::function<NativeDynamicShapePlan*()>& factory,
    bool acquireLease) {

  std::lock_guard<std::mutex> shutdownLock(shutdownMutex_);
  if (isShutdownInProgress()) return nullptr;

  NativeDynamicShapePlan* result = nullptr;
  std::vector<NativeDynamicShapePlan*> victims;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Cache hit: splice to front (MRU), optionally acquire a borrower lease,
    // and return. Same-borrower redispatch passes acquireLease=false.
    auto it = map_.find(key);
    if (it != map_.end()) {
      lru_.splice(lru_.begin(), lru_, it->second);
      NativeDynamicShapePlan* plan = it->second->second;
      if (acquireLease) ++pinCounts_[plan];
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
      logPlanCacheMemoryState("HIT_PINNED", &key, plan, lru_.size(), pinCounts_.size());
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

      // Insert at MRU (front) and acquire the initial borrower lease. A
      // cache miss always belongs to the caller even when acquireLease=false.
      lru_.emplace_front(key, plan);
      map_[key] = lru_.begin();
      pinCounts_[plan] = 1;
      logPlanCacheMemoryState(
          "INSERT_BEFORE_BUDGET", &key, plan, lru_.size(), pinCounts_.size());

      // Enforce count and memory budgets (skips pinned plans).
      // Victims are returned, NOT deleted under the lock.
      victims = evictIfOverBudgetLocked();
      logPlanCacheMemoryState(
          "INSERT_AFTER_BUDGET", &key, plan, lru_.size(), pinCounts_.size());

      result = plan;
    }
  }  // mutex_ released here

  // Release and delete evicted plans OUTSIDE the mutex. GPU teardown makes
  // CUDA API calls and may migrate the final frozen owner of a shared weight
  // buffer, so it must not run under the cache lock. Release all victims before
  // deleting any of them for the same last-owner ordering used by clear().
  for (auto* victim : victims) {
    if (victim != nullptr) {
      victim->releaseGpuIntermediates();
    }
  }
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
  bool clearAfterRelease = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pinCounts_.find(plan);
    if (it == pinCounts_.end()) {
      logPlanCacheMemoryState("UNPIN_NOT_PINNED", nullptr, plan,
                              lru_.size(), pinCounts_.size());
      return;
    }
    if (it->second > 1) {
      --it->second;
      logPlanCacheMemoryState("UNPIN_LEASE_RELEASED", nullptr, plan,
                              lru_.size(), pinCounts_.size());
    } else {
      pinCounts_.erase(it);
      clearAfterRelease = clearPending_;
      logPlanCacheMemoryState("UNPIN", nullptr, plan,
                              lru_.size(), pinCounts_.size());
    }
  }

  // A prior clear intentionally retained leased entries. Once this release
  // reaches zero, rerun the safe clear path so the entry cannot be reused.
  if (clearAfterRelease) clear();
}

// ---------------------------------------------------------------------------
// pinnedCount
// ---------------------------------------------------------------------------

size_t NativePlanCache::pinnedCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pinCounts_.size();
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
  const size_t pinnedPlanEstimate = dspHasDeviceMemory()
      ? kBytesPerPlanEstimate +
            static_cast<size_t>(std::max(0, dsp.captureWorkspaceMb())) *
                1024ULL * 1024ULL +
            static_cast<size_t>(std::max(
                0, sd::Environment::getInstance().dspCublasWorkspaceMb())) *
                1024ULL * 1024ULL
      : kBytesPerPlanEstimate;
  logPlanCacheMemoryState(
      "BUDGET_CHECK", nullptr, nullptr, lru_.size(), pinCounts_.size());

  // Helper lambda: find oldest unpinned, non-passivated plan (LRU end toward MRU front)
  auto findPassivationCandidate = [&]() -> LruList::iterator {
    for (auto it = std::prev(lru_.end()); ; ) {
      if (pinCounts_.count(it->second) == 0 && !it->second->isPassivated()) {
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
      if (pinCounts_.count(it->second) == 0) {
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
      // A leased plan can be mutating its slot/handle vectors on another
      // execution thread. It cannot be evicted anyway, so charge a conservative
      // configured workspace estimate without dereferencing mutable plan state.
      if (pinCounts_.count(entry.second) != 0) {
        total += pinnedPlanEstimate;
        continue;
      }
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
    if (victim == lru_.end()) {
      DSP_DIAG(MEMORY,
               "PLAN_CACHE_BUDGET_BLOCKED reason=count_cap_all_pinned entries=%zu "
               "pinned=%zu maxPlans=%d",
               lru_.size(), pinCounts_.size(), maxPlans);
      break;
    }
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
        if (candidate == lru_.end()) {
          DSP_DIAG(MEMORY,
                   "PLAN_CACHE_BUDGET_BLOCKED reason=memory_all_pinned_or_passivated "
                   "entries=%zu pinned=%zu cache=%zuMB budget=%zuMB",
                   lru_.size(), pinCounts_.size(),
                   totalCacheBytes / (1024 * 1024),
                   budgetBytes / (1024 * 1024));
          break;
        }

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
        if (victim == lru_.end()) {
          DSP_DIAG(MEMORY,
                   "PLAN_CACHE_BUDGET_BLOCKED reason=eviction_all_pinned "
                   "entries=%zu pinned=%zu cache=%zuMB budget=%zuMB",
                   lru_.size(), pinCounts_.size(),
                   totalCacheBytes / (1024 * 1024),
                   budgetBytes / (1024 * 1024));
          break;
        }

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
