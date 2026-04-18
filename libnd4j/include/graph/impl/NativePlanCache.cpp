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
#include <helpers/logger.h>
#include <system/Environment.h>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

namespace sd {
namespace graph {

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
    return plan;
  }

  // Cache miss: build a new plan.
  NativeDynamicShapePlan* plan = factory();
  if (!plan) {
    return nullptr;
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

  const int maxPlans   = dsp.planCacheMaxPlans();
  const float fraction = dsp.planCacheBudgetFraction();

  // ---- 1. Hard count cap ----
  // Iterate from LRU (back) toward MRU (front), skipping pinned plans.
  while (static_cast<int>(lru_.size()) > maxPlans) {
    // Find the oldest unpinned plan.
    auto victim = lru_.end();
    for (auto it = std::prev(lru_.end()); ; ) {
      if (pinnedPlans_.count(it->second) == 0) {
        victim = it;
        break;
      }
      if (it == lru_.begin()) break;
      --it;
    }
    if (victim == lru_.end()) {
      // All plans are pinned — cannot evict further.
      break;
    }
    sd_printf("[NativePlanCache] evict LRU plan (count cap %d): outputSetHash=%llu phPtrs=%zu\n",
              maxPlans,
              (unsigned long long)victim->first.outputSetHash,
              victim->first.phShapeInfoPtrs.size());
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
    if (err != cudaSuccess) {
      // If we can't query free memory, skip the soft cap silently.
      freeMem = 0;
    }
#endif

    if (freeMem > 0) {
      const size_t budgetBytes = static_cast<size_t>(fraction * static_cast<float>(freeMem));

      while (lru_.size() * kBytesPerPlanEstimate > budgetBytes && !lru_.empty()) {
        // Find the oldest unpinned plan.
        auto victim = lru_.end();
        for (auto it = std::prev(lru_.end()); ; ) {
          if (pinnedPlans_.count(it->second) == 0) {
            victim = it;
            break;
          }
          if (it == lru_.begin()) break;
          --it;
        }
        if (victim == lru_.end()) {
          // All remaining plans are pinned — cannot evict further.
          break;
        }
        sd_printf("[NativePlanCache] evict LRU plan (memory budget %.1f%% of %zuMB free): "
                  "outputSetHash=%llu phPtrs=%zu\n",
                  fraction * 100.0f,
                  freeMem / (1024 * 1024),
                  (unsigned long long)victim->first.outputSetHash,
                  victim->first.phShapeInfoPtrs.size());
        map_.erase(victim->first);
        delete victim->second;
        lru_.erase(victim);
      }
    }
  }
}

}  // namespace graph
}  // namespace sd
