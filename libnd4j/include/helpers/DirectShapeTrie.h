#ifndef DIRECTSHAPETRIE_H
#define DIRECTSHAPETRIE_H

#include <array/ConstantShapeBuffer.h>
#include <system/common.h>

#include <array>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace sd {

#if __cplusplus >= 201703L
#define SHAPE_MUTEX_TYPE std::shared_mutex
#else
#define SHAPE_MUTEX_TYPE std::mutex
#endif

class SD_LIB_EXPORT ShapeTrieNode {
 private:
  std::vector<std::unique_ptr<ShapeTrieNode>> _children;
  LongType _value;
  int _level;
  bool _isShape;
  ConstantShapeBuffer* _buffer;  // Now accessed atomically
  int _shapeHash;   // Store a hash of the shape for validation
  
#if defined(SD_GCC_FUNCTRACE)
  backward::StackTrace storeStackTrace;
#endif

 public:
  ShapeTrieNode(LongType value = 0, int level = 0, bool isShape = true, int shapeHash = 0)
      : _value(value), _level(level), _isShape(isShape), _buffer(nullptr), _shapeHash(shapeHash) {}

  ~ShapeTrieNode() = default;

  ShapeTrieNode* findOrCreateChild(LongType value, int level, bool isShape, int shapeHash = 0) {
    for (auto& child : _children) {
      if (child->value() == value && child->isShape() == isShape && child->shapeHash() == shapeHash) {
        return child.get();
      }
    }
    
    auto newNode = std::make_unique<ShapeTrieNode>(value, level, isShape, shapeHash);
    auto* ptr = newNode.get();
    _children.push_back(std::move(newNode));
    return ptr;
  }

  const std::vector<std::unique_ptr<ShapeTrieNode>>& children() const { return _children; }
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
  std::array<std::unique_ptr<ShapeTrieNode>, NUM_STRIPES> _roots;
  mutable std::array<SHAPE_MUTEX_TYPE, NUM_STRIPES> _mutexes = {};

 public:
  // Constructor
  DirectShapeTrie() {
    for (size_t i = 0; i < NUM_STRIPES; i++) {
      _roots[i] = std::make_unique<ShapeTrieNode>(0, 0, false);
      // Make sure mutexes are properly initialized
      new (&_mutexes[i]) SHAPE_MUTEX_TYPE();  // Explicit initialization
    }
  }

  // Delete copy constructor and assignment
  DirectShapeTrie(const DirectShapeTrie&) = delete;
  DirectShapeTrie& operator=(const DirectShapeTrie&) = delete;

  // Delete move operations
  DirectShapeTrie(DirectShapeTrie&&) = delete;
  DirectShapeTrie& operator=(DirectShapeTrie&&) = delete;

  // Create a shape buffer from shapeInfo
  ConstantShapeBuffer* createBuffer(const LongType* shapeInfo);

  // Improved thread-safe getOrCreate
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