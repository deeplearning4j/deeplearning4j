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
  TadPack* _tadPack;  // Now accessed atomically
  int _shapeRank;     // Store the rank of the original shape for verification

 public:
  TadTrieNode(LongType value = 0, int level = 0, bool isDimension = true, int shapeRank = 0)
      : _value(value), _level(level), _isDimension(isDimension), _tadPack(nullptr), _shapeRank(shapeRank) {}

  ~TadTrieNode() = default;

  TadTrieNode* findOrCreateChild(LongType value, int level, bool isDimension, int shapeRank = 0) {
    for (auto& child : _children) {
      if (child->value() == value && child->isDimension() == isDimension && child->shapeRank() == shapeRank) {
        return child.get();
      }
    }
#ifndef __JAVACPP_HACK__
    auto newNode = std::make_unique<TadTrieNode>(value, level, isDimension, shapeRank);
    auto* ptr = newNode.get();
    _children.push_back(std::move(newNode));
#endif
    return ptr;
  }

  const std::vector<std::unique_ptr<TadTrieNode>>& children() const { return _children; }
  LongType value() const { return _value; }
  int level() const { return _level; }
  bool isDimension() const { return _isDimension; }
  int shapeRank() const { return _shapeRank; }

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
  static const size_t NUM_STRIPES = 128; // Increased from 32 to reduce collision chance
  std::array<std::unique_ptr<TadTrieNode>, NUM_STRIPES> _roots;
  mutable std::array<TAD_MUTEX_TYPE, NUM_STRIPES> _mutexes = {};
  std::array<std::atomic<int>, NUM_STRIPES> _stripeCounts = {};

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

  // Enhanced getOrCreate with improved thread safety
  TadPack* getOrCreate(std::vector<LongType>& dimensions, LongType* originalShape);

  // Original methods preserved
  size_t computeStripeIndex(const std::vector<LongType>& dimensions, LongType* originalShape) const {
    size_t hash = 17; // Prime number starting point

    // Add dimension-specific hash contribution with position-dependence
    for (size_t i = 0; i < dimensions.size(); i++) {
      hash = hash * 31 + static_cast<size_t>(dimensions[i]) * (i + 1);
    }

    // Add rank - critical for distinguishing different dimension arrays
    int rank = shape::rank(originalShape);
    hash = hash * 13 + rank * 19;

    // Add shape signature based on shape dimensions with position-dependence
    LongType* shapeInfo = shape::shapeOf(originalShape);
    for (int i = 0; i < rank; i++) {
      hash = hash * 17 + static_cast<size_t>(shapeInfo[i]) * (11 + i);
    }

    // Add total element count to distinguish differently sized arrays
    hash = hash * 41 + shape::length(originalShape);

    return hash % NUM_STRIPES;
  }

  // Calculate comprehensive shape hash for node identification
  size_t calculateShapeHash(LongType* originalShape) const {
    size_t hash = 17;

    int rank = shape::rank(originalShape);
    hash = hash * 31 + rank * 13;

    LongType* shapeInfo = shape::shapeOf(originalShape);
    for (int i = 0; i < rank; i++) {
      hash = hash * 19 + static_cast<size_t>(shapeInfo[i]) * (7 + i);
    }

    LongType* strides = shape::stride(originalShape);
    for (int i = 0; i < rank; i++) {
      hash = hash * 23 + static_cast<size_t>(strides[i]) * (11 + i);
    }

    // Add data type and order
    hash = hash * 29 + static_cast<size_t>(ArrayOptions::dataType(originalShape));
    hash = hash * 37 + static_cast<size_t>(shape::order(originalShape));

    return hash;
  }


  bool exists(const std::vector<LongType>& dimensions, LongType* originalShape) const;

  // Original helper methods preserved
  TadPack* search(const std::vector<LongType>& dimensions, int originalShapeRank, size_t stripeIdx) const;
  std::vector<LongType> sortDimensions(const std::vector<LongType>& dimensions) const;
  bool dimensionsEqual(const std::vector<LongType>& a, const std::vector<LongType>& b) const;
  const TadTrieNode* findChild(const TadTrieNode* node, LongType value, int level, bool isDimension, int shapeRank) const;
  TadPack* insert(std::vector<LongType>& dimensions, LongType* originalShape);
};

}  // namespace sd
#endif
#endif //LIBND4J_DIRECTTADTRIE_H