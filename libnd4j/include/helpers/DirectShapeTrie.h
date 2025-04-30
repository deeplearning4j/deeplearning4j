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
#include <memory>
#include <vector>

#if __cplusplus >= 201703L && (!defined(__APPLE__))
#include <shared_mutex>

#else
#include <mutex>
#endif
namespace sd {

#if defined(__APPLE__)
#include <Availability.h>
#endif

#if __cplusplus >= 201703L && (!defined(__APPLE__))
#include <shared_mutex>
#define SHAPE_MUTEX_TYPE std::shared_mutex
#define SHAPE_LOCK_TYPE std::shared_lock
#else
#include <mutex>
#define SHAPE_MUTEX_TYPE std::mutex
#define SHAPE_LOCK_TYPE std::lock_guard
#endif

class SD_LIB_EXPORT ShapeTrieNode {
 private:
  std::vector<ShapeTrieNode*> _children;
  LongType _value;
  int _level;
  bool _isShape;
  ConstantShapeBuffer* _buffer = nullptr;  // Now accessed atomically
  int _shapeHash;   // Store a hash of the shape for validation

#if defined(SD_GCC_FUNCTRACE)
  backward::StackTrace storeStackTrace;
#endif

 public:
  ShapeTrieNode(LongType value = 0, int level = 0, bool isShape = true, sd::LongType shapeHash = 0)
      : _value(value), _level(level), _isShape(isShape), _shapeHash(shapeHash) {
  }

  ~ShapeTrieNode() {
    // Delete children
    for (auto* child : _children) {
      delete child;
    }
    _children.clear();

    // Delete buffer if it exists
    if (_buffer != nullptr) {
      delete _buffer;
      _buffer = nullptr;
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
  ConstantShapeBuffer* buffer() const { return _buffer; }

#if defined(SD_GCC_FUNCTRACE)
  void collectStoreStackTrace();
#endif
};

class SD_LIB_EXPORT DirectShapeTrie {
 private:
  static const size_t NUM_STRIPES = 256; // Increased from 32 to reduce collisions
  std::array<ShapeTrieNode*, NUM_STRIPES> *_roots;
  std::array<SHAPE_MUTEX_TYPE*, NUM_STRIPES> *_mutexes = nullptr;

  // Helper method to create a fallback buffer when trie insertion fails
  // Always returns a valid shape buffer or throws an exception
  ConstantShapeBuffer* createFallbackBuffer(const LongType* shapeInfo, int rank);

 public:
  // Constructor
  DirectShapeTrie() {
    _roots = new std::array<ShapeTrieNode*, NUM_STRIPES>();
    _mutexes = new std::array<SHAPE_MUTEX_TYPE*, NUM_STRIPES>();

    for (size_t i = 0; i < NUM_STRIPES; i++) {
      (*_roots)[i] = new ShapeTrieNode(0, 0, false);
      // Allocate mutexes on the heap
      (*_mutexes)[i] = new SHAPE_MUTEX_TYPE();
    }

    ShapeBufferPlatformHelper::initialize();
  }

  // Delete copy constructor and assignment
  DirectShapeTrie(const DirectShapeTrie&) = delete;
  DirectShapeTrie& operator=(const DirectShapeTrie&) = delete;

  ~DirectShapeTrie() {
    // Clean up all mutexes
    if (_mutexes != nullptr) {
      for (size_t i = 0; i < NUM_STRIPES; i++) {
        if ((*_mutexes)[i] != nullptr) {
          delete (*_mutexes)[i];
        }
      }
      delete _mutexes;
    }

    // Clean up all root nodes
    if (_roots != nullptr) {
      for (size_t i = 0; i < NUM_STRIPES; i++) {
        if ((*_roots)[i] != nullptr) {
          delete (*_roots)[i];
        }
      }
      delete _roots;
    }
  }

  // Delete move operations
  DirectShapeTrie(DirectShapeTrie&&) = delete;
  DirectShapeTrie& operator=(DirectShapeTrie&&) = delete;

  // Improved thread-safe getOrCreate
  // Always returns a valid shape buffer or throws an exception
  ConstantShapeBuffer* getOrCreate(const LongType* shapeInfo);

  // Check if a shape info already exists in the trie
  bool exists(const LongType* shapeInfo) const;

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
};

}  // namespace sd

#endif  // DIRECTSHAPETRIE_H