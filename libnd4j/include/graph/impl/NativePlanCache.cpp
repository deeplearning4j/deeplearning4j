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
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <memory/MemoryUtils.h>
#include <system/Environment.h>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

namespace sd {
namespace graph {

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
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : lru_) {
    delete entry.second;
  }
  lru_.clear();
  map_.clear();
  pinnedPlans_.clear();
}

// ---------------------------------------------------------------------------
// getOrInsert
// ---------------------------------------------------------------------------

NativeDynamicShapePlan* NativePlanCache::getOrInsert(
    const Key& key,
    const std::function<NativeDynamicShapePlan*()>& factory) {

  std::lock_guard<std::mutex> lock(mutex_);

  // Cache hit: splice to front (MRU), pin, and return.
  auto it = map_.find(key);
  if (it != map_.end()) {
    lru_.splice(lru_.begin(), lru_, it->second);
    NativeDynamicShapePlan* plan = it->second->second;
    pinnedPlans_.insert(plan);
    DSP_DIAG(EXECUTE, "PLAN_CACHE HIT thread=0x%llx plan=%p shapeHash=0x%llx",
             (unsigned long long)key.threadId, (void*)plan,
             (unsigned long long)key.phShapeContentHash);
    return plan;
  }

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
  evictIfOverBudgetLocked();

  return plan;
}

// ---------------------------------------------------------------------------
// unpinPlan
// ---------------------------------------------------------------------------

void NativePlanCache::unpinPlan(NativeDynamicShapePlan* plan) {
  if (!plan) return;
  std::lock_guard<std::mutex> lock(mutex_);
  pinnedPlans_.erase(plan);
  evictIfOverBudgetLocked();
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

void NativePlanCache::evictIfOverBudgetLocked() {
  auto& dsp = sd::Environment::getInstance().dsp();

  // Select the correct hard cap based on build type
#ifdef SD_CUDA
  const int maxPlans = dsp.planCacheMaxPlans();
#else
  const int maxPlans = dsp.planCacheMaxPlansCpu();
#endif
  const float fraction = dsp.planCacheBudgetFraction();

  // Helper lambda: find oldest unpinned plan (LRU end toward MRU front)
  auto findVictim = [&]() -> LruList::iterator {
    for (auto it = std::prev(lru_.end()); ; ) {
      if (pinnedPlans_.count(it->second) == 0) {
        return it;
      }
      if (it == lru_.begin()) break;
      --it;
    }
    return lru_.end();
  };

  // ---- 1. Hard count cap ----
  while (static_cast<int>(lru_.size()) > maxPlans) {
    auto victim = findVictim();
    if (victim == lru_.end()) break;  // all pinned
    DSP_DIAG(MEMORY, "PLAN_CACHE evict LRU (count cap %d): outputSetHash=%llu phCount=%lld contentHash=0x%016llx",
             maxPlans,
             (unsigned long long)victim->first.outputSetHash,
             (long long)victim->first.phCount,
             (unsigned long long)victim->first.phShapeContentHash);
    map_.erase(victim->first);
    delete victim->second;
    lru_.erase(victim);
  }

  // ---- 2. Memory-fraction soft cap ----
  if (fraction > 0.0f && !lru_.empty()) {
    size_t freeMem = 0;

#ifdef SD_CUDA
    size_t totalMem = 0;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) freeMem = 0;
#else
    freeMem = sd::memory::MemoryUtils::getSystemFreeMemoryBytes();
#endif

    if (freeMem > 0) {
      const size_t budgetBytes = static_cast<size_t>(fraction * static_cast<float>(freeMem));

      // Compute actual cache footprint using per-plan estimates
      size_t totalCacheBytes = 0;
      for (auto& entry : lru_) {
        size_t planBytes = entry.second->estimatedOwnedBytes();
        totalCacheBytes += (planBytes > 0) ? planBytes : kBytesPerPlanEstimate;
      }

      while (totalCacheBytes > budgetBytes && !lru_.empty()) {
        auto victim = findVictim();
        if (victim == lru_.end()) break;  // all pinned

        size_t victimBytes = victim->second->estimatedOwnedBytes();
        size_t victimCost = (victimBytes > 0) ? victimBytes : kBytesPerPlanEstimate;

        DSP_DIAG(MEMORY, "PLAN_CACHE evict LRU (memory budget %.1f%% of %zuMB free, cache=%zuMB): "
                 "outputSetHash=%llu phCount=%lld contentHash=0x%016llx planBytes=%zuMB",
                 fraction * 100.0f,
                 freeMem / (1024 * 1024),
                 totalCacheBytes / (1024 * 1024),
                 (unsigned long long)victim->first.outputSetHash,
                 (long long)victim->first.phCount,
                 (unsigned long long)victim->first.phShapeContentHash,
                 victimCost / (1024 * 1024));
        totalCacheBytes -= victimCost;
        map_.erase(victim->first);
        delete victim->second;
        lru_.erase(victim);
      }
    }
  }
}

}  // namespace graph
}  // namespace sd
