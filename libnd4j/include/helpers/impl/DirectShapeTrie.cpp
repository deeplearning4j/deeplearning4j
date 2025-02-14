#include <helpers/DirectShapeTrie.h>
#include <array/ConstantShapeBuffer.h>
#include <array/PrimaryPointerDeallocator.h>
#include <array/DataType.h>
#include <array/ArrayOptions.h>
#include <helpers/shape.h>
#include <system/common.h>

namespace sd {

thread_local DirectShapeTrie::ThreadCache DirectShapeTrie::_threadCache;

void ShapeTrieNode::setBuffer(ConstantShapeBuffer* buf) {
  ConstantShapeBuffer* old = _buffer;
  _buffer = buf;
  if (old) delete old;
}

#if defined(SD_GCC_FUNCTRACE)
void ShapeTrieNode::collectStoreStackTrace() {
  this->storeStackTrace = backward::StackTrace();
  this->storeStackTrace.load_here(32);
}
#endif

size_t DirectShapeTrie::computeHash(const LongType* shapeInfo) const {
  size_t hash = 14695981039346656037ULL;
  const size_t len = shape::shapeInfoLength(shape::rank(shapeInfo));

  size_t i = 0;
  for (; i + 4 <= len; i += 4) {
    hash ^= static_cast<size_t>(shapeInfo[i]);
    hash *= 1099511628211ULL;
    hash ^= static_cast<size_t>(shapeInfo[i + 1]);
    hash *= 1099511628211ULL;
    hash ^= static_cast<size_t>(shapeInfo[i + 2]);
    hash *= 1099511628211ULL;
    hash ^= static_cast<size_t>(shapeInfo[i + 3]);
    hash *= 1099511628211ULL;
  }

  for (; i < len; i++) {
    hash ^= static_cast<size_t>(shapeInfo[i]);
    hash *= 1099511628211ULL;
  }

  return hash;
}

size_t DirectShapeTrie::getStripeIndex(const LongType* shapeInfo) const {
  return computeHash(shapeInfo) & (NUM_STRIPES - 1);
}

bool DirectShapeTrie::shapeInfoEqual(const LongType* a, const LongType* b) const {
  if (a == b) return true;
  if (a == nullptr || b == nullptr) return false;

  const int rankA = shape::rank(a);
  if (rankA != shape::rank(b)) return false;

  const int len = shape::shapeInfoLength(rankA);
  return std::memcmp(a, b, len * sizeof(LongType)) == 0;
}

void DirectShapeTrie::validateShapeInfo(const LongType* shapeInfo) const {
  if (shapeInfo == nullptr) {
    THROW_EXCEPTION("Shape info cannot be null");
  }

  const int rank = shape::rank(shapeInfo);
  if (rank < 0 || rank > SD_MAX_RANK) {
    std::string errorMessage = "Invalid rank: " + std::to_string(rank) +
                               ". Valid range is 0 to " + std::to_string(SD_MAX_RANK);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (rank == 0) {
    const int len = shape::shapeInfoLength(rank);
    bool allZero = true;
    for (int i = 0; i < len; i++) {
      if (shapeInfo[i] != 0) {
        allZero = false;
        break;
      }
    }
    if (allZero) {
      THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
    }
  }

  if (ArrayOptions::dataType(shapeInfo) == UNKNOWN) {
    THROW_EXCEPTION("Shape info created with invalid data type");
  }

  char order = shape::order(shapeInfo);
  if (order != 'c' && order != 'f') {
    std::string errorMessage = "Invalid ordering in shape buffer: ";
    errorMessage += order;
    THROW_EXCEPTION(errorMessage.c_str());
  }
}

ConstantShapeBuffer* DirectShapeTrie::createBuffer(const LongType* shapeInfo) {
  const int shapeInfoLength = shape::shapeInfoLength(shape::rank(shapeInfo));
  LongType* shapeCopy = new LongType[shapeInfoLength];
  std::memcpy(shapeCopy, shapeInfo, shapeInfoLength * sizeof(LongType));

  auto deallocator = std::shared_ptr<PrimaryPointerDeallocator>(new PrimaryPointerDeallocator(), [] (PrimaryPointerDeallocator* ptr) { delete ptr; });
  auto hPtr = std::make_shared<PointerWrapper>(shapeCopy, deallocator);
  return new ConstantShapeBuffer(hPtr);
}

void DirectShapeTrie::updateThreadCache(const LongType* shapeInfo, ConstantShapeBuffer* buffer) {
  auto& cache = _threadCache.entries;
  if (cache.size() >= ThreadCache::CACHE_SIZE) {
    cache.erase(cache.begin());
  }
  cache.emplace_back(shapeInfo, buffer);
}

const ShapeTrieNode* DirectShapeTrie::findChild(const ShapeTrieNode* node, LongType value,
                                                int level, bool isShape) const {
  if (!node) return nullptr;

  for (const auto& child : node->children()) {
    if (child->value() == value &&
        child->level() == level &&
        child->isShape() == isShape) {
      return child.get();
    }
  }
  return nullptr;
}

ConstantShapeBuffer* DirectShapeTrie::search(const LongType* shapeInfo, size_t stripeIdx) const {
  // No locks here - caller handles locking
  const ShapeTrieNode* current = _roots[stripeIdx].get();
  const int rank = shape::rank(shapeInfo);

  // Check rank
  current = findChild(current, rank, 0, true);
  if (!current) return nullptr;

  // Check datatype
  current = findChild(current, ArrayOptions::dataType(shapeInfo), 1, true);
  if (!current) return nullptr;

  // Check order
  current = findChild(current, shape::order(shapeInfo), 2, true);
  if (!current) return nullptr;

  // Check shape values
  const LongType* shape = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = findChild(current, shape[i], 3 + i, true);
    if (!current) return nullptr;
  }

  // Check stride values
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = findChild(current, strides[i], 3 + rank + i, false);
    if (!current) return nullptr;
  }

  return current ? current->buffer() : nullptr;
}



ConstantShapeBuffer* DirectShapeTrie::insert(const LongType* shapeInfo, size_t stripeIdx) {
  ShapeTrieNode* current = _roots[stripeIdx].get();
  const int rank = shape::rank(shapeInfo);

  // Insert rank
  current = current->findOrCreateChild(rank, 0, true);
  if (!current) return nullptr;

  // Insert datatype
  current = current->findOrCreateChild(ArrayOptions::dataType(shapeInfo), 1, true);
  if (!current) return nullptr;

  // Insert order
  current = current->findOrCreateChild(shape::order(shapeInfo), 2, true);
  if (!current) return nullptr;

  // Insert shape values
  const LongType* shape = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(shape[i], 3 + i, true);
    if (!current) return nullptr;
  }

  // Insert stride values
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(strides[i], 3 + rank + i, false);
    if (!current) return nullptr;
  }

  if (!current->buffer()) {
    try {
      const int shapeInfoLength = shape::shapeInfoLength(rank);
      LongType* shapeCopy = new LongType[shapeInfoLength];
      std::memcpy(shapeCopy, shapeInfo, shapeInfoLength * sizeof(LongType));

      auto deallocator = std::shared_ptr<PrimaryPointerDeallocator>(new PrimaryPointerDeallocator(),
                                                                    [] (PrimaryPointerDeallocator* ptr) { delete ptr; });
      auto hPtr = std::make_shared<PointerWrapper>(shapeCopy, deallocator);
      auto buffer = new ConstantShapeBuffer(hPtr);

      current->setBuffer(buffer);
      return buffer;
    } catch (const std::exception& e) {
      std::string msg = "Shape buffer creation failed: ";
      msg += e.what();
      THROW_EXCEPTION(msg.c_str());
    }
  }

  return current->buffer();
}


bool DirectShapeTrie::exists(const LongType* shapeInfo) const {
  validateShapeInfo(shapeInfo);
  size_t stripeIdx = getStripeIndex(shapeInfo);

  std::unique_lock<SHAPE_MUTEX_TYPE> lock(_mutexes[stripeIdx]);
  return search(shapeInfo, stripeIdx) != nullptr;
}



ConstantShapeBuffer* DirectShapeTrie::getOrCreate(const LongType* shapeInfo) {
  validateShapeInfo(shapeInfo);
  size_t stripeIdx = getStripeIndex(shapeInfo);

  // Single lock pattern, matching TAD trie
  std::unique_lock<SHAPE_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

  // Check if it exists
  if (auto buffer = search(shapeInfo, stripeIdx)) {
    return buffer;
  }

  // If not found, create it
  return insert(shapeInfo, stripeIdx);
}

}  // namespace sd