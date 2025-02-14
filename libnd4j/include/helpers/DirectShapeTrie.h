#ifndef LIBND4J_DIRECTSHAPETRIE_H
#define LIBND4J_DIRECTSHAPETRIE_H

#include <system/common.h>
#include <array/ConstantShapeBuffer.h>

#include <array>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace sd {
#ifndef __JAVACPP_HACK__


class SD_LIB_EXPORT ShapeTrieNode {
 private:
  std::vector<std::unique_ptr<ShapeTrieNode>> _children;
  LongType _value;
  int _level;
  bool _isShape;
  ConstantShapeBuffer* _buffer;  // Changed from atomic

#if defined(SD_GCC_FUNCTRACE)
  backward::StackTrace st;
  backward::StackTrace storeStackTrace;
#endif

 public:
  ShapeTrieNode(LongType value = 0, int level = 0, bool isShape = true)
      : _value(value), _level(level), _isShape(isShape), _buffer(nullptr) {
#if defined(SD_GCC_FUNCTRACE)
    this->st.load_here();
#endif
  }

  ~ShapeTrieNode() {
    if (_buffer) delete _buffer;
  }

  ShapeTrieNode* findOrCreateChild(LongType value, int level, bool isShape) {
    for (auto& child : _children) {
      if (child->value() == value && child->level() == level &&
          child->isShape() == isShape) {
        return child.get();
      }
    }

    auto newNode = std::make_unique<ShapeTrieNode>(value, level, isShape);
    auto* ptr = newNode.get();
    _children.push_back(std::move(newNode));
    return ptr;
  }

  const std::vector<std::unique_ptr<ShapeTrieNode>>& children() const { return _children; }
  LongType value() const { return _value; }
  int level() const { return _level; }
  bool isShape() const { return _isShape; }

  void setBuffer(ConstantShapeBuffer* buf);
  ConstantShapeBuffer* buffer() const { return _buffer; }

#if defined(SD_GCC_FUNCTRACE)
  void collectStoreStackTrace();
#endif
};

#if __cplusplus >= 201703L
#define SHAPE_MUTEX_TYPE std::shared_mutex
#define SHAPE_LOCK_TYPE std::shared_lock
#else
#define SHAPE_MUTEX_TYPE std::mutex
#define SHAPE_LOCK_TYPE std::lock_guard
#endif

class SD_LIB_EXPORT DirectShapeTrie {
 private:
  static const size_t NUM_STRIPES = 32;
  std::array<std::unique_ptr<ShapeTrieNode>, NUM_STRIPES> _roots;
  mutable std::array<SHAPE_MUTEX_TYPE, NUM_STRIPES> _mutexes = {};
  std::array<std::atomic<int>, NUM_STRIPES> _stripeCounts = {};

  // Enhanced thread-local cache with atomic operations
  struct alignas(64) ThreadCache {
    static const size_t CACHE_SIZE = 1024;
    std::vector<std::pair<const LongType*, ConstantShapeBuffer*>> entries;
    std::atomic<size_t> size{0};

    ThreadCache() {
      entries.reserve(CACHE_SIZE);
    }
  };

  static thread_local ThreadCache _threadCache;

  size_t computeHash(const LongType* shapeInfo) const;
  size_t getStripeIndex(const LongType* shapeInfo) const;
  bool shapeInfoEqual(const LongType* a, const LongType* b) const;
  void validateShapeInfo(const LongType* shapeInfo) const;
  ConstantShapeBuffer* createBuffer(const LongType* shapeInfo);
  void updateThreadCache(const LongType* shapeInfo, ConstantShapeBuffer* buffer);
  const ShapeTrieNode* findChild(const ShapeTrieNode* node, LongType value,
                                 int level, bool isShape) const;
  ConstantShapeBuffer* search(const LongType* shapeInfo, size_t stripeIdx) const;
  ConstantShapeBuffer* insert(const LongType* shapeInfo, size_t stripeIdx);

 public:
  // Constructor
  DirectShapeTrie() {
#ifndef __JAVACPP_HACK__
    for (size_t i = 0; i < NUM_STRIPES; i++) {
      _roots[i] = std::make_unique<ShapeTrieNode>(0, 0, false);
      // Make sure mutexes are properly initialized
      new (&_mutexes[i]) SHAPE_MUTEX_TYPE();
    }
#endif
  }

  // Delete copy constructor and assignment
  DirectShapeTrie(const DirectShapeTrie&) = delete;
  DirectShapeTrie& operator=(const DirectShapeTrie&) = delete;

  // Delete move operations
  DirectShapeTrie(DirectShapeTrie&&) = delete;
  DirectShapeTrie& operator=(DirectShapeTrie&&) = delete;

  // Enhanced getOrCreate with three-tier access pattern
  ConstantShapeBuffer* getOrCreate(const LongType* shapeInfo);
  bool exists(const LongType* shapeInfo) const;
};

}  // namespace sd
#endif
#endif //LIBND4J_DIRECTSHAPETRIE_H