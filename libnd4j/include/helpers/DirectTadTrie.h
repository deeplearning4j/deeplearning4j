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

#ifndef LIBND4J_DIRECTTADTRIE_H
#define LIBND4J_DIRECTTADTRIE_H

#include <array/TadPack.h>
#include <system/common.h>

#include <array>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "array/TadCalculator.h"

namespace sd {
#ifndef __JAVACPP_HACK__

// Add make_unique implementation for pre-C++14
class SD_LIB_EXPORT TadTrieNode {
 private:
  std::vector<std::unique_ptr<TadTrieNode>> _children;
  LongType _value;
  int _level;
  bool _isDimension;
  TadPack* _tadPack;  // Changed from atomic

 public:
  TadTrieNode(LongType value = 0, int level = 0, bool isDimension = true)
      : _value(value), _level(level), _isDimension(isDimension), _tadPack(nullptr) {}

  ~TadTrieNode() = default;

  TadTrieNode* findOrCreateChild(LongType value, int level, bool isDimension) {
    for (auto& child : _children) {
      if (child->value() == value && child->isDimension() == isDimension) {
        return child.get();
      }
    }
#ifndef __JAVACPP_HACK__
    auto newNode = std::make_unique<TadTrieNode>(value, level, isDimension);
    auto* ptr = newNode.get();
    _children.push_back(std::move(newNode));
#endif
    return ptr;
  }

  const std::vector<std::unique_ptr<TadTrieNode>>& children() const { return _children; }
  LongType value() const { return _value; }
  int level() const { return _level; }
  bool isDimension() const { return _isDimension; }

  void setPack(TadPack* pack);

  TadPack* pack() const { return _tadPack; }
};

#if __cplusplus >= 201703L
#define TAD_MUTEX_TYPE std::shared_mutex
#define TAD_LOCK_TYPE std::shared_lock
#else
#define TAD_MUTEX_TYPE std::mutex
#define TAD_LOCK_TYPE std::lock_guard
#endif

class SD_LIB_EXPORT DirectTadTrie {
 private:
  static const size_t NUM_STRIPES = 32;
  std::array<std::unique_ptr<TadTrieNode>, NUM_STRIPES> _roots;
  mutable std::array<TAD_MUTEX_TYPE, NUM_STRIPES> _mutexes = {};
  std::array<std::atomic<int>, NUM_STRIPES> _stripeCounts = {};

  // Enhanced thread-local cache with atomic operations
  struct alignas(64) ThreadCache {
    static const size_t CACHE_SIZE = 1024;
    std::vector<std::pair<std::vector<LongType>, TadPack*>> entries;
    std::atomic<size_t> size{0};

    ThreadCache() {
      entries.reserve(CACHE_SIZE);
    }
  };

  static thread_local ThreadCache _threadCache;

 public:
  // Constructor
  DirectTadTrie() {
#ifndef __JAVACPP_HACK__
    for (size_t i = 0; i < NUM_STRIPES; i++) {
      _roots[i] = std::make_unique<TadTrieNode>(0, 0, false);
      // Make sure mutexes are properly initialized
      new (&_mutexes[i]) TAD_MUTEX_TYPE();  // Explicit initialization
    }
#endif
  }

  // Delete copy constructor and assignment
  DirectTadTrie(const DirectTadTrie&) = delete;
  DirectTadTrie& operator=(const DirectTadTrie&) = delete;

  // Delete move operations
  DirectTadTrie(DirectTadTrie&&) = delete;
  DirectTadTrie& operator=(DirectTadTrie&&) = delete;

  // Enhanced getOrCreate with three-tier access pattern
  TadPack* getOrCreate(std::vector<LongType>& dimensions, LongType* originalShape) {
    const size_t stripeIdx = computeStripeIndex(dimensions);
    std::unique_lock<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

    // Traverse to the correct node or create path as needed
    TadTrieNode* current = _roots[stripeIdx].get();

    // First level: dimension length
    current = current->findOrCreateChild(dimensions.size(), 0, false);
    if (!current) {
      THROW_EXCEPTION("Failed to create/find length node");
    }

    // Second level: dimensions
    for (size_t i = 0; i < dimensions.size(); i++) {
      current = current->findOrCreateChild(dimensions[i], i + 1, true);
      if (!current) {
        THROW_EXCEPTION("Failed to create/find dimension node");
      }
    }

    // Check if we already have a TAD pack
    if (TadPack* existing = current->pack()) {
      return existing;
    }

    // Create new TAD pack under the same lock
    try {
      TadCalculator calculator(originalShape);
      calculator.createTadPack(dimensions);

      TadPack* newPack = new TadPack(
          calculator.tadShape(),
          calculator.tadOffsets(),
          calculator.numberOfTads(),
          dimensions.data(),
          dimensions.size());

      current->setPack(newPack);
      return newPack;
    } catch (const std::exception& e) {
      std::string msg = "TAD creation failed: ";
      msg += e.what();
      THROW_EXCEPTION(msg.c_str());
    }
  }

  // Original methods preserved
  size_t computeStripeIndex(const std::vector<LongType>& dimensions) const {
    size_t hash = 0;
    for (auto dim : dimensions) {
      hash = hash * 31 + static_cast<size_t>(dim);
    }
    return hash % NUM_STRIPES;
  }

  bool exists(const std::vector<LongType>& dimensions) const ;

  // Original helper methods preserved
  TadPack* search(const std::vector<LongType>& dimensions, size_t stripeIdx) const;
  std::vector<LongType> sortDimensions(const std::vector<LongType>& dimensions) const;
  bool dimensionsEqual(const std::vector<LongType>& a, const std::vector<LongType>& b) const;
  const TadTrieNode* findChild(const TadTrieNode* node, LongType value, int level, bool isDimension) const;
  TadPack* insert(std::vector<LongType>& dimensions, LongType* originalShape);
};

}  // namespace sd
#endif
#endif //LIBND4J_DIRECTTADTRIE_H