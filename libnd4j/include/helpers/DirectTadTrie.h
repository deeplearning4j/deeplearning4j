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
#include <unordered_set>
#include "./generic/StripedLocks.h"

#include <vector>

#include "array/TadCalculator.h"

// Add include for MSVC compiler intrinsics
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace sd {
#ifndef __JAVACPP_HACK__

/**
 * Stores cached metadata about a TadPack for fast comparison without recomputation.
 * Matches only on shape, rank, and dataType (not strides or order).
 */
struct TadPackSignature {
  LongType* shape = nullptr;
  int rank = 0;
  DataType dataType = DataType::FLOAT32;

  ~TadPackSignature() {
    if (shape) delete[] shape;
  }

  // Store signature from shapeInfo (FIXED: No longer stores strides/order)
  void store(LongType* shapeInfo) {
    if (!shapeInfo) return;

    rank = shape::rank(shapeInfo);
    dataType = ArrayOptions::dataType(shapeInfo);

    // Allocate and copy shape only (strides removed)
    if (shape) delete[] shape;
    shape = new LongType[rank];
    LongType* srcShape = shape::shapeOf(shapeInfo);
    for (int i = 0; i < rank; i++) {
      shape[i] = srcShape[i];
    }
  }

  // Compare with another shapeInfo (FIXED: No longer compares strides/order)
  bool matches(LongType* shapeInfo) const {
    if (!shapeInfo || !shape) return false;

    int otherRank = shape::rank(shapeInfo);
    if (rank != otherRank) return false;


    if (dataType != ArrayOptions::dataType(shapeInfo)) return false;

    LongType* otherShape = shape::shapeOf(shapeInfo);
    for (int i = 0; i < rank; i++) {
      if (shape[i] != otherShape[i]) return false;
    }

    return true;
  }
};

class SD_LIB_EXPORT TadTrieNode {
 private:
  std::vector<std::unique_ptr<TadTrieNode>> _children;
  LongType _value;
  int _level;
  bool _isDimension;
  std::shared_ptr<TadPack> _tadPack;  // Use shared_ptr to prevent deletion while in use
  int _shapeRank;     // Store the rank of the original shape for verification
  size_t _nodeHash;   // Additional hash for quick comparison
  TadPackSignature* _packSignature;  // Cached signature for fast comparison

 public:
  TadTrieNode(LongType value = 0, int level = 0, bool isDimension = true, int shapeRank = 0)
      : _value(value), _level(level), _isDimension(isDimension), _tadPack(nullptr),
        _shapeRank(shapeRank), _nodeHash(0), _packSignature(nullptr) {}

  // Delete copy operations to prevent issues with unique_ptr
  TadTrieNode(const TadTrieNode&) = delete;
  TadTrieNode& operator=(const TadTrieNode&) = delete;

  // Enable move operations
  TadTrieNode(TadTrieNode&&) = default;
  TadTrieNode& operator=(TadTrieNode&&) = default;

  ~TadTrieNode() {
    if (_packSignature) {
      delete _packSignature;
      _packSignature = nullptr;
    }
    // _tadPack is now shared_ptr - it cleans up automatically via RAII
    // No manual delete needed
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
   // NOTE: Removed #ifndef __JAVACPP_HACK__ guard to fix TAD cache memory leak
   // The guard was preventing proper ownership chain when JavaCPP is used (production mode)
   // Without this, child nodes were never added to _children vector, preventing automatic
   // cleanup when parent nodes are destroyed, causing 100% TAD pack leak rate
   auto newNode = std::make_unique<TadTrieNode>(value, level, isDimension, shapeRank);
   newNode->_nodeHash = childHash;
   auto* ptr = newNode.get();
   _children.push_back(std::move(newNode));
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

 // Standard getters
 const std::vector<std::unique_ptr<TadTrieNode>>& children() const { return _children; }
 LongType value() const { return _value; }
 int level() const { return _level; }
 bool isDimension() const { return _isDimension; }
 int shapeRank() const { return _shapeRank; }
 size_t nodeHash() const { return _nodeHash; }
 const TadPackSignature* packSignature() const { return _packSignature; }

 // Enhanced TadPack setter with signature caching
 void setPack(std::shared_ptr<TadPack> pack) {
   // Thread-safe assignment using atomic operations
   std::atomic_store(&_tadPack, pack);

   // Cache the signature for future fast comparisons
   if (pack && pack->primaryShapeInfo() && !_packSignature) {
     _packSignature = new TadPackSignature();
     _packSignature->store(pack->primaryShapeInfo());
   }
 }

 std::shared_ptr<TadPack> pack() const {
   return std::atomic_load(&_tadPack);
 }
};



class SD_LIB_EXPORT DirectTadTrie {
private:
 static const size_t NUM_STRIPES = 128; // Increased from 32 to reduce collision chance
 std::array<std::unique_ptr<TadTrieNode>, NUM_STRIPES> _roots;
 mutable std::array<MUTEX_TYPE, NUM_STRIPES> _mutexes = {};
 std::array<std::atomic<int>, NUM_STRIPES> _stripeCounts = {};

 // Cache statistics tracking
 mutable std::atomic<LongType> _current_entries{0};
 mutable std::atomic<LongType> _current_bytes{0};
 mutable std::atomic<LongType> _peak_entries{0};
 mutable std::atomic<LongType> _peak_bytes{0};

 // Internal helper to recursively count entries and bytes in a subtrie
 void countEntriesAndBytes(const TadTrieNode* node, LongType& entries, LongType& bytes) const;

public:
 // Constructor
 DirectTadTrie() {
   // NOTE: Removed #ifndef __JAVACPP_HACK__ guard to fix TAD cache memory leak
   // Without proper initialization, roots remain nullptr causing crashes
   // or preventing proper cache management
   for (size_t i = 0; i < NUM_STRIPES; i++) {
     _roots[i] = std::make_unique<TadTrieNode>(0, 0, false);
     // Make sure mutexes are properly initialized
     new (&_mutexes[i]) MUTEX_TYPE();  // Explicit initialization
   }
 }

 // Destructor - clean up all cached TAD packs on singleton destruction
 ~DirectTadTrie() {
   // Clear all TAD packs to prevent memory leaks
   // This is called when the ConstantTadHelper singleton is destroyed on JVM shutdown
   clear();
 }

 // Delete copy constructor and assignment
 DirectTadTrie(const DirectTadTrie&) = delete;
 DirectTadTrie& operator=(const DirectTadTrie&) = delete;

 // Delete move operations
 DirectTadTrie(DirectTadTrie&&) = delete;
 DirectTadTrie& operator=(DirectTadTrie&&) = delete;

 std::shared_ptr<TadPack> enhancedSearch(const std::vector<LongType>& dimensions, LongType* originalShape, size_t stripeIdx);
 bool exists(const std::vector<LongType>& dimensions, LongType* originalShape);
 // Enhanced stride-aware hash computation
 size_t computeStrideAwareHash(const std::vector<LongType>& dimensions, LongType* originalShape) ;

 // Enhanced getOrCreate with improved thread safety
 std::shared_ptr<TadPack> getOrCreate(std::vector<LongType>& dimensions, LongType* originalShape);

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

 bool exists(const std::vector<LongType>& dimensions, LongType* originalShape) const;

 // Original helper methods preserved
 std::shared_ptr<TadPack> search(const std::vector<LongType>& dimensions, int originalShapeRank, size_t stripeIdx) const;
 std::vector<LongType> sortDimensions(const std::vector<LongType>& dimensions) const;
 const TadTrieNode* findChild(const TadTrieNode* node, LongType value, int level, bool isDimension, int shapeRank) const;
 std::shared_ptr<TadPack> insert(std::vector<LongType>& dimensions, LongType* originalShape);

 /**
  * Clear all cached TAD packs to prevent memory leaks during testing.
  * This recreates the root nodes, which will delete all child nodes and their TadPacks.
  */
 void clear();

 /**
  * Get the total number of cached TAD pack entries.
  *
  * @return Total number of cached TAD packs across all stripes
  */
 LongType getCachedEntries() const;

 /**
  * Get the total memory used by cached TAD packs in bytes.
  * This includes both shape_info and offset buffer sizes.
  *
  * @return Total memory used in bytes
  */
 LongType getCachedBytes() const;

 /**
  * Get the peak number of TAD pack entries that were cached simultaneously.
  *
  * @return Peak number of cached TAD packs
  */
 LongType getPeakCachedEntries() const;

 /**
  * Get the peak memory usage by cached TAD packs in bytes.
  *
  * @return Peak memory usage in bytes
  */
 LongType getPeakCachedBytes() const;

 /**
  * Generate a human-readable string representation of the trie structure.
  * Shows the hierarchy of nodes and cached TAD packs for debugging.
  *
  * @param maxDepth Maximum depth to traverse (default: 10, -1 for unlimited)
  * @param maxEntries Maximum number of entries to show (default: 100, -1 for unlimited)
  * @return String representation of the trie
  */
 std::string toString(int maxDepth = 10, int maxEntries = 100) const;

 /**
  * Get all TadPack pointers currently in the cache.
  * This is used by lifecycle tracking to distinguish cached entries from real leaks.
  *
  * @param out_pointers Set to fill with pointers to all cached TadPack objects
  */
 void getCachedPointers(std::unordered_set<void*>& out_pointers) const;

private:
 // Internal helper to build string representation recursively
 void buildStringRepresentation(const TadTrieNode* node, std::stringstream& ss,
                                const std::string& indent, int currentDepth,
                                int maxDepth, int& entriesShown, int maxEntries) const;

 // Internal helper to collect TadPack pointers recursively
 void collectCachedPointers(const TadTrieNode* node, std::unordered_set<void*>& out_pointers) const;
};

}  // namespace sd
#endif
#endif //LIBND4J_DIRECTTADTRIE_H
