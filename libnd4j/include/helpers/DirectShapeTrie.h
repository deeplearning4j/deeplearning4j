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

#ifndef DIRECTSHAPETRIE_H
#define DIRECTSHAPETRIE_H

#include <array/ConstantShapeBuffer.h>
#include <system/common.h>
#include "ShapeBufferPlatformHelper.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <unordered_set>
#include <vector>
#include "./generic/StripedLocks.h"
namespace sd {



class SD_LIB_EXPORT ShapeTrieNode {
 private:
  std::vector<ShapeTrieNode*> _children;
  LongType _value;
  int _level;
  bool _isShape;
  std::atomic<ConstantShapeBuffer*> _buffer{nullptr};
  int _shapeHash;   // Store a hash of the shape for validation

#if defined(SD_GCC_FUNCTRACE)
  backward::StackTrace storeStackTrace;
#endif

 public:
  ShapeTrieNode(LongType value = 0, int level = 0, bool isShape = true, sd::LongType shapeHash = 0)
      : _value(value), _level(level), _isShape(isShape), _shapeHash(shapeHash) {
  }

  // Padded operator new/delete to protect adjacent glibc chunks from
  // overruns. ShapeTrieNode objects are small (~80 bytes) and heavily
  // allocated during shape trie population — any adjacent overrun
  // corrupts the next chunk metadata → SIGABRT on free().
  static void* operator new(size_t size) {
    return std::malloc(size + 4096);
  }
#ifndef __JAVACPP_HACK__
  static void* operator new(size_t size, const std::nothrow_t& tag) noexcept {
    return std::malloc(size + 4096);
  }
#endif
  static void operator delete(void* ptr) noexcept {
    std::free(ptr);
  }
#ifndef __JAVACPP_HACK__
  static void operator delete(void* ptr, const std::nothrow_t& tag) noexcept {
    std::free(ptr);
  }
#endif

  ~ShapeTrieNode() {
    // Delete children
    for (auto* child : _children) {
      delete child;
    }
    _children.clear();

    ConstantShapeBuffer* buf = _buffer.exchange(nullptr, std::memory_order_acq_rel);

    if (buf != nullptr) {
      buf->release();
    }
  }


  // Replace the current findOrCreateChild with this more defensive version
  ShapeTrieNode* findOrCreateChild(LongType value, int level, bool isShape, int shapeHash) {
    // First search for existing child
    for (auto* child : _children) {
      if (child != nullptr &&
          child->value() == value &&
          child->level() == level &&
          child->isShape() == isShape &&
          (shapeHash == 0 || child->shapeHash() == shapeHash)) {
        return child;
      }
    }

    // Create new node with standard allocation
    try {
      auto* newNode = new ShapeTrieNode(value, level, isShape, shapeHash);
      // Add to children using standard vector operations
      _children.push_back(newNode);
      return newNode;
    } catch (const std::exception& e) {
      return nullptr;
    }
  }

  const std::vector<ShapeTrieNode*>& children() const { return _children; }
  LongType value() const { return _value; }
  int level() const { return _level; }
  bool isShape() const { return _isShape; }
  int shapeHash() const { return _shapeHash; }

  void setBuffer(ConstantShapeBuffer* buf);
  // Thread-safe atomic read of the buffer pointer
  ConstantShapeBuffer* buffer() const { return _buffer.load(std::memory_order_acquire); }

#if defined(SD_GCC_FUNCTRACE)
  void collectStoreStackTrace();
#endif
};

class SD_LIB_EXPORT DirectShapeTrie {
 private:
  static const size_t NUM_STRIPES = 256; // Increased from 32 to reduce collisions
  std::array<ShapeTrieNode*, NUM_STRIPES> *_roots;
  std::array<MUTEX_TYPE*, NUM_STRIPES> *_mutexes = nullptr;

  // Initialization guards to prevent partially-constructed state from leaking
  std::atomic<bool> _initialization_complete{false};
  std::atomic<bool> _initialization_in_progress{false};

  // Cache statistics tracking
  mutable std::atomic<LongType> _current_entries{0};
  mutable std::atomic<LongType> _current_bytes{0};
  mutable std::atomic<LongType> _peak_entries{0};
  mutable std::atomic<LongType> _peak_bytes{0};

  // Shutdown flag to prevent cache clearing during JVM shutdown
  // When set, clearCache() becomes a no-op to avoid segfaults
  std::atomic<bool> _shutdownInProgress{false};

  // Helper method to create a fallback buffer when trie insertion fails
  // Always returns a valid shape buffer or throws an exception
  ConstantShapeBuffer* createFallbackBuffer(const LongType* shapeInfo, int rank);

  // Internal helper to recursively count entries and bytes in a subtrie
  void countEntriesAndBytes(const ShapeTrieNode* node, LongType& entries, LongType& bytes) const;

  // Internal helper to build string representation recursively
  void buildStringRepresentation(const ShapeTrieNode* node, std::stringstream& ss,
                                 const std::string& indent, int currentDepth,
                                 int maxDepth, int& entriesShown, int maxEntries) const;

 public:
  // Constructor
  DirectShapeTrie() {
    _initialization_in_progress.store(true, std::memory_order_release);
    _roots = new std::array<ShapeTrieNode*, NUM_STRIPES>();
    _mutexes = new std::array<MUTEX_TYPE*, NUM_STRIPES>();

    for (size_t i = 0; i < NUM_STRIPES; i++) {
      (*_roots)[i] = new ShapeTrieNode(0, 0, false);
      // Allocate mutexes on the heap
      (*_mutexes)[i] = new MUTEX_TYPE();
    }

    ShapeBufferPlatformHelper::initialize();
    _initialization_in_progress.store(false, std::memory_order_release);
    _initialization_complete.store(true, std::memory_order_release);
  }

  // Delete copy constructor and assignment
  DirectShapeTrie(const DirectShapeTrie&) = delete;
  DirectShapeTrie& operator=(const DirectShapeTrie&) = delete;

  // Destructor - Intentionally leak memory to prevent crashes during JVM shutdown.
  // During static destruction, memory allocators may already be destroyed, causing
  // corrupted pointers. Traversing the tree to delete nodes causes SIGSEGV.
  // The OS reclaims all memory on process exit anyway.
  // For explicit cleanup during runtime (e.g., testing), use clear() directly.
  ~DirectShapeTrie() {
    // Intentionally do NOT delete anything here.
    // Setting pointers to nullptr prevents double-free but we don't traverse the tree.
    _mutexes = nullptr;
    _roots = nullptr;
  }

  // Delete move operations
  DirectShapeTrie(DirectShapeTrie&&) = delete;
  DirectShapeTrie& operator=(DirectShapeTrie&&) = delete;

  // Improved thread-safe getOrCreate
  // Always returns a valid shape buffer or throws an exception
  ConstantShapeBuffer* getOrCreate(const LongType* shapeInfo);

  // Check if a shape info already exists in the trie
  bool exists(const LongType* shapeInfo) const;

  // Wait until constructor finishes allocating internal state.
  void waitForInitialization() const;

  // Helper methods
  size_t computeHash(const LongType* shapeInfo) const;
  size_t getStripeIndex(const LongType* shapeInfo) const;
  bool shapeInfoEqual(const LongType* a, const LongType* b) const;
  void validateShapeInfo(const LongType* shapeInfo) const;
  ConstantShapeBuffer* search(const LongType* shapeInfo, size_t stripeIdx) const;
  ConstantShapeBuffer* insert(const LongType* shapeInfo, size_t stripeIdx);
  const ShapeTrieNode* findChild(const ShapeTrieNode* node, LongType value, int level, bool isShape, int shapeHash = 0) const;

  // Calculate a unique shape signature for additional validation
  int calculateShapeSignature(const LongType* shapeInfo) const;

  // Clear all cached shape buffers
  // NOTE: Will return early without action if setShutdownInProgress(true) was called.
  void clearCache();

  /**
   * Mark that JVM/application shutdown is in progress.
   * When true, clearCache() becomes a no-op to avoid segfaults during static destruction.
   * The OS will reclaim all memory at process exit anyway.
   *
   * @param inProgress true to prevent cache clearing, false to re-enable
   */
  void setShutdownInProgress(bool inProgress) {
    _shutdownInProgress.store(inProgress, std::memory_order_release);
  }

  /**
   * Check if shutdown is in progress.
   *
   * @return true if setShutdownInProgress(true) was called
   */
  bool isShutdownInProgress() const {
    return _shutdownInProgress.load(std::memory_order_acquire);
  }

  /**
   * Get the total number of cached shape entries.
   *
   * @return Total number of cached shape buffers across all stripes
   */
  LongType getCachedEntries() const;

  /**
   * Get the total memory used by cached shape buffers in bytes.
   * This includes the shape_info buffer sizes across all cached entries.
   *
   * @return Total memory used in bytes
   */
  LongType getCachedBytes() const;

  /**
   * Get the peak number of entries that were cached simultaneously.
   *
   * @return Peak number of cached entries
   */
  LongType getPeakCachedEntries() const;

  /**
   * Get the peak memory usage by cached shape buffers in bytes.
   *
   * @return Peak memory usage in bytes
   */
  LongType getPeakCachedBytes() const;

  /**
   * Generate a human-readable string representation of the trie structure.
   * Shows the hierarchy of nodes and cached shape buffers for debugging.
   *
   * @param maxDepth Maximum depth to traverse (default: 10, -1 for unlimited)
   * @param maxEntries Maximum number of entries to show (default: 100, -1 for unlimited)
   * @return String representation of the trie
   */
  std::string toString(int maxDepth = 10, int maxEntries = 100) const;

  /**
   * Get all ConstantShapeBuffer pointers currently in the cache.
   * This is used by lifecycle tracking to distinguish cached entries from real leaks.
   *
   * @param out_pointers Set to fill with pointers to all cached ConstantShapeBuffer objects
   */
  void getCachedPointers(std::unordered_set<void*>& out_pointers) const;

 private:
  // Internal helper to collect ConstantShapeBuffer pointers recursively
  void collectCachedPointers(const ShapeTrieNode* node, std::unordered_set<void*>& out_pointers) const;
};

}  // namespace sd

#endif  // DIRECTSHAPETRIE_H
