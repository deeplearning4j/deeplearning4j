/* ******************************************************************************
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
* See the NOTICE file distributed with this work for additional
* information regarding copyright ownership.
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
#include "./generic/StripedLocks.h"

#include <vector>

#include "array/TadCalculator.h"

// Add include for MSVC compiler intrinsics
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace sd {
#ifndef __JAVACPP_HACK__

class SD_LIB_EXPORT TadTrieNode {
 private:
  std::vector<std::unique_ptr<TadTrieNode>> _children;
  LongType _value;
  int _level;
  bool _isDimension;
  TadPack* _tadPack;  // Accessed atomically
  int _shapeRank;     // Store the rank of the original shape for verification
  size_t _nodeHash;   // Additional hash for quick comparison
  LongType* _originalStrides; // Store the original strides signature (optional)
  int _stridesLength;  // Length of the strides array

 public:
  TadTrieNode(LongType value = 0, int level = 0, bool isDimension = true, int shapeRank = 0)
      : _value(value), _level(level), _isDimension(isDimension), _tadPack(nullptr),
        _shapeRank(shapeRank), _nodeHash(0), _originalStrides(nullptr), _stridesLength(0) {}

  // Delete copy operations to prevent issues with unique_ptr
  TadTrieNode(const TadTrieNode&) = delete;
  TadTrieNode& operator=(const TadTrieNode&) = delete;

  // Enable move operations
  TadTrieNode(TadTrieNode&&) = default;
  TadTrieNode& operator=(TadTrieNode&&) = default;

  ~TadTrieNode() {
    // Clean up original strides if allocated
    if (_originalStrides != nullptr) {
      delete[] _originalStrides;
      _originalStrides = nullptr;
    }
  }


 // Improved child finding with hash-based optimization
 TadTrieNode* findOrCreateChild(LongType value, int level, bool isDimension, int shapeRank = 0) {
   // First check using hash-based comparison if available
   size_t childHash = computeChildHash(value, isDimension, shapeRank);

   for (auto& child : _children) {
     if (child->_nodeHash == childHash) {
       // Fast path: hash match, do detailed comparison
       if (child->value() == value &&
           child->isDimension() == isDimension &&
           child->shapeRank() == shapeRank) {
         return child.get();
       }
     }
   }

   // Not found, create new child
#ifndef __JAVACPP_HACK__
   auto newNode = std::make_unique<TadTrieNode>(value, level, isDimension, shapeRank);
   newNode->_nodeHash = childHash;
   auto* ptr = newNode.get();
   _children.push_back(std::move(newNode));
#endif
   return ptr;
 }

 // Compute hash for faster child comparison
 size_t computeChildHash(LongType value, bool isDimension, int shapeRank) const {
   size_t hash = 17;
   hash = hash * 31 + static_cast<size_t>(value);
   hash = hash * 13 + (isDimension ? 1 : 0);
   hash = hash * 7 + static_cast<size_t>(shapeRank);
   return hash;
 }

 // Method to store stride information
 void storeStrides(LongType* strides, int length) {
   if (_originalStrides != nullptr) {
     delete[] _originalStrides;
   }

   if (strides && length > 0) {
     _originalStrides = new LongType[length];
     _stridesLength = length;

     // Copy strides
     for (int i = 0; i < length; i++) {
       _originalStrides[i] = strides[i];
     }
   }
 }


 // Standard getters
 const std::vector<std::unique_ptr<TadTrieNode>>& children() const { return _children; }
 LongType value() const { return _value; }
 int level() const { return _level; }
 bool isDimension() const { return _isDimension; }
 int shapeRank() const { return _shapeRank; }

 size_t nodeHash() const { return _nodeHash; }



 // Enhanced TadPack setter with stride storage
 void setPack(TadPack* pack) {
   if (!pack) return;

   // Use atomic compare-and-swap for thread safety
   TadPack* expectedNull = nullptr;
   bool swapped = false;

#if defined(_MSC_VER)
   // MSVC-specific atomic operation
   swapped = (_InterlockedCompareExchangePointer(reinterpret_cast<void* volatile*>(&_tadPack), pack, expectedNull) == expectedNull);
#else
   // GCC/Clang-specific atomic operation
   swapped = __sync_bool_compare_and_swap(&_tadPack, expectedNull, pack);
#endif

   if (swapped) {
     // Successfully set the pack
     // Store stride information for future comparisons
     if (pack->primaryShapeInfo()) {
       int rank = shape::rank(pack->primaryShapeInfo());
       LongType* strides = shape::stride(pack->primaryShapeInfo());
       storeStrides(strides, rank);
     }
     // The new pack is now owned by the trie, do nothing.
   } else if (pack != _tadPack) {
     // If the swap failed, another thread set the pack.
     // We must delete the pack we tried to insert to avoid a memory leak.
     delete pack;
   }
 }

 TadPack* pack() const { return _tadPack; }

 // Removed operator== to avoid unique_ptr copy issues
 // If equality comparison is needed, implement it without comparing _children vector directly

};



class SD_LIB_EXPORT DirectTadTrie {
private:
 static const size_t NUM_STRIPES = 128; // Increased from 32 to reduce collision chance
 std::array<std::unique_ptr<TadTrieNode>, NUM_STRIPES> _roots;
 mutable std::array<MUTEX_TYPE, NUM_STRIPES> _mutexes = {};
 std::array<std::atomic<int>, NUM_STRIPES> _stripeCounts = {};

public:
 // Constructor
 DirectTadTrie() {
#ifndef __JAVACPP_HACK__
   for (size_t i = 0; i < NUM_STRIPES; i++) {
     _roots[i] = std::make_unique<TadTrieNode>(0, 0, false);
     // Make sure mutexes are properly initialized
     new (&_mutexes[i]) MUTEX_TYPE();  // Explicit initialization
   }
#endif
 }

 // Delete copy constructor and assignment
 DirectTadTrie(const DirectTadTrie&) = delete;
 DirectTadTrie& operator=(const DirectTadTrie&) = delete;

 // Delete move operations
 DirectTadTrie(DirectTadTrie&&) = delete;
 DirectTadTrie& operator=(DirectTadTrie&&) = delete;

 TadPack* enhancedSearch(const std::vector<LongType>& dimensions, LongType* originalShape, size_t stripeIdx);
 bool exists(const std::vector<LongType>& dimensions, LongType* originalShape);
 // Enhanced stride-aware hash computation
 size_t computeStrideAwareHash(const std::vector<LongType>& dimensions, LongType* originalShape) ;

 // Dimension compatibility check
 bool areDimensionsCompatible(const std::vector<LongType>& dimensions, const TadPack* pack) ;
 // Enhanced getOrCreate with improved thread safety
 TadPack* getOrCreate(std::vector<LongType>& dimensions, LongType* originalShape);

 // Original methods preserved
 size_t computeStripeIndex(const std::vector<LongType>& dimensions, LongType* originalShape) const {
   size_t hash = 17; // Prime number starting point

   // Add dimension-specific hash contribution with position-dependence
   for (size_t i = 0; i < dimensions.size(); i++) {
     hash = hash * 31 + static_cast<size_t>(dimensions[i]) * (i + 1);
   }

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
 const TadTrieNode* findChild(const TadTrieNode* node, LongType value, int level, bool isDimension, int shapeRank) const;
 TadPack* insert(std::vector<LongType>& dimensions, LongType* originalShape);
};

}  // namespace sd
#endif
#endif //LIBND4J_DIRECTTADTRIE_H
