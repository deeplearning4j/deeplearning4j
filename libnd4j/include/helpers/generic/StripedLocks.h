/* ******************************************************************************
*
* Copyright (c) 2024 Konduit K.K.
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef SD_STRIPED_LOCKS_H_
#define SD_STRIPED_LOCKS_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "system/op_boilerplate.h"

namespace sd {
namespace generic {

template<size_t NUM_STRIPES = 32>
class StripedLocks {
private:
 mutable std::array<std::shared_mutex, NUM_STRIPES> _mutexes;
 mutable std::array<std::atomic<uint32_t>, NUM_STRIPES> _stripeCounts{};
 static constexpr int MAX_RETRIES = 500;
 static constexpr auto RETRY_DELAY = std::chrono::microseconds(50);
 static constexpr int MAX_BACKOFF_SHIFT = 10;
 static constexpr auto MIN_DELAY = std::chrono::microseconds(50);

 static_assert((NUM_STRIPES & (NUM_STRIPES - 1)) == 0, "NUM_STRIPES must be a power of 2");

 bool acquireLockWithTimeout(size_t stripe, bool exclusive, int maxRetries) const {
   auto startTime = std::chrono::steady_clock::now();

   for (int attempt = 0; attempt < maxRetries; ++attempt) {
     bool locked = exclusive ?
                             _mutexes[stripe].try_lock() :
                             _mutexes[stripe].try_lock_shared();

     if (locked) {
       _stripeCounts[stripe].fetch_add(1, std::memory_order_acq_rel);
       return true;
     }

     if (std::chrono::steady_clock::now() - startTime > std::chrono::seconds(1)) {
       return false;
     }

     auto backoff = RETRY_DELAY * (1 << std::min(attempt, MAX_BACKOFF_SHIFT));
     std::this_thread::sleep_for(backoff);
   }
   return false;
 }

public:
 StripedLocks() {
   initializeCounts();
 }

 class MultiLockGuard {
  private:
   StripedLocks& _locks;
   std::vector<size_t> _stripes;
   bool _exclusive;
   bool _acquired{false};
   static constexpr auto LOCK_TIMEOUT = std::chrono::seconds(2);

   bool acquireAllLocksWithTimeout(const std::chrono::milliseconds& timeout) {
     auto startTime = std::chrono::steady_clock::now();
     std::vector<size_t> acquired;
     acquired.reserve(_stripes.size());

     while (std::chrono::steady_clock::now() - startTime < timeout) {
       bool allLocked = true;
       for (size_t stripe : _stripes) {
         if (!_locks.acquireLockWithTimeout(stripe, _exclusive, 1)) {
           allLocked = false;
           for (auto& s : acquired) {
             _locks.unlockStripe(s, _exclusive);
           }
           acquired.clear();
           break;
         }
         acquired.push_back(stripe);
       }
       if (allLocked) {
         _acquired = true;
         return true;
       }
       std::this_thread::sleep_for(MIN_DELAY);
     }
     return false;
   }

  public:
   MultiLockGuard(StripedLocks& locks,
                  const std::vector<size_t>& stripes,
                  bool exclusive,
                  const std::chrono::milliseconds& timeout = std::chrono::seconds(2))
       : _locks(locks), _stripes(stripes), _exclusive(exclusive) {
     std::sort(_stripes.begin(), _stripes.end());
     _stripes.erase(std::unique(_stripes.begin(), _stripes.end()), _stripes.end());
     acquireAllLocksWithTimeout(timeout);
   }

   bool acquired() const { return _acquired; }

   void release() {
     if (!_acquired) return;
     for (auto it = _stripes.rbegin(); it != _stripes.rend(); ++it) {
       _locks.unlockStripe(*it, _exclusive);
     }
     _acquired = false;
   }

   ~MultiLockGuard() {
     if (_acquired) {
       release();
     }
   }

   MultiLockGuard(const MultiLockGuard&) = delete;
   MultiLockGuard& operator=(const MultiLockGuard&) = delete;
   MultiLockGuard(MultiLockGuard&&) noexcept = default;
 };

 MultiLockGuard acquireMultiLockWithTimeout(const std::vector<size_t>& stripes,
                                            bool exclusive,
                                            const std::chrono::milliseconds& timeout) {
   return MultiLockGuard(*this, stripes, exclusive, timeout);
 }

 void lockStripe(size_t stripe, bool exclusive = false) const {
   if (stripe >= NUM_STRIPES) {
     throw std::out_of_range("Invalid stripe index");
   }

   if (!acquireLockWithTimeout(stripe, exclusive, MAX_RETRIES)) {
     auto currentCount = _stripeCounts[stripe].load(std::memory_order_relaxed);
     std::string msg = "Failed to acquire " +
                       std::string(exclusive ? "exclusive" : "shared") +
                       " lock for stripe " + std::to_string(stripe) +
                       " after " + std::to_string(MAX_RETRIES) +
                       " attempts. Current count: " + std::to_string(currentCount);
     THROW_EXCEPTION(msg.c_str());
   }
 }

 void unlockStripe(size_t stripe, bool exclusive = false) const {
   if (stripe >= NUM_STRIPES) {
     throw std::out_of_range("Invalid stripe index");
   }
   _stripeCounts[stripe].fetch_sub(1, std::memory_order_relaxed);
   if (exclusive) {
     _mutexes[stripe].unlock();
   } else {
     _mutexes[stripe].unlock_shared();
   }
 }

 MultiLockGuard acquireMultiLock(const std::vector<size_t>& stripes, bool exclusive = false) {
   return MultiLockGuard(*this, stripes, exclusive);
 }

 uint32_t getStripeCount(size_t stripe) const {
   return _stripeCounts[stripe].load(std::memory_order_relaxed);
 }

 template<typename T>
 size_t getStripeIndex(const T& value) const {
   return std::hash<T>{}(value) & (NUM_STRIPES - 1);
 }

 void initializeCounts() {
   for (auto& count : _stripeCounts) {
     count.store(0, std::memory_order_relaxed);
   }
 }

 static constexpr size_t getNumStripes() { return NUM_STRIPES; }
};

}} // namespace sd::generic

#endif // SD_STRIPED_LOCKS_H_