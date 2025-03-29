#include <array/ArrayOptions.h>
#include <array/ConstantShapeBuffer.h>
#include <array/DataType.h>
#include <array/PrimaryPointerDeallocator.h>
#include <helpers/DirectShapeTrie.h>
#include <helpers/shape.h>
#include <system/common.h>

#include <atomic>
#include <memory>

#include "helpers/ShapeBufferCreatorHelper.h"

namespace sd {

void ShapeTrieNode::setBuffer(ConstantShapeBuffer* buf) {
  if (!buf) return;

  // Use atomic compare-and-swap for thread safety
  ConstantShapeBuffer* expectedNull = nullptr;
  if (_buffer == nullptr &&
      __sync_bool_compare_and_swap(&_buffer, expectedNull, buf)) {
    // Successfully set the buffer when it was null
    return;
  } else if (_buffer != nullptr) {
    // Buffer is already set - DO NOTHING
    // Just keep using the existing buffer

    // IMPORTANT: Only delete the new buffer if we're sure it's safe to do so
    // Check if the shapes are actually the same before deleting
    if (buf != _buffer) {
      if (shape::equalsSoft(buf->primary(), _buffer->primary())) {
        // The shapes match, safe to delete the duplicate
        // delete buf;
      } else {
        // The shapes don't match - this is an error condition!
        // We should log this and NOT delete the buffer as it may be used elsewhere
        printf("WARNING: Attempted to set different shape buffer on same node!\n");
        // Don't delete - it might be used elsewhere
      }
    }
  }
}

#if defined(SD_GCC_FUNCTRACE)
void ShapeTrieNode::collectStoreStackTrace() {
  this->storeStackTrace = backward::StackTrace();
  this->storeStackTrace.load_here(32);
}
#endif

size_t DirectShapeTrie::computeHash(const LongType* shapeInfo) const {
  size_t hash = 17; // Prime number starting point
  const int rank = shape::rank(shapeInfo);

  // Add rank first with high weight
  hash = hash * 31 + rank * 19;

  // Add shape elements to hash with position-dependent multipliers
  const LongType* shape = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    hash = hash * 13 + static_cast<size_t>(shape[i]) * (7 + i);
  }

  // Add stride elements to hash with position-dependent multipliers
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    hash = hash * 19 + static_cast<size_t>(strides[i]) * (11 + i);
  }

  // Add data type and order with higher weights
  hash = hash * 23 + static_cast<size_t>(ArrayOptions::dataType(shapeInfo)) * 29;
  hash = hash * 37 + static_cast<size_t>(shape::order(shapeInfo)) * 41;

  // Add total element count
  hash = hash * 43 + shape::length(shapeInfo);

  return hash;
}

int DirectShapeTrie::calculateShapeSignature(const LongType* shapeInfo) const {
  int signature = 17;
  const int rank = shape::rank(shapeInfo);

  // Incorporate rank with weight
  signature = signature * 31 + rank * 13;

  // Incorporate shape dimensions with position weights
  const LongType* shapeValues = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    signature = signature * 13 + static_cast<int>(shapeValues[i]) * (7 + i);
  }

  // Incorporate data type and order
  signature = signature * 7 + static_cast<int>(ArrayOptions::dataType(shapeInfo)) * 11;
  signature = signature * 17 + static_cast<int>(shape::order(shapeInfo)) * 19;

  // Include element count
  signature = signature * 23 + static_cast<int>(shape::length(shapeInfo) % 10000);

  return signature;
}

size_t DirectShapeTrie::getStripeIndex(const LongType* shapeInfo) const {
  return computeHash(shapeInfo) % NUM_STRIPES;
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

const ShapeTrieNode* DirectShapeTrie::findChild(const ShapeTrieNode* node, LongType value,
                                                int level, bool isShape, int shapeHash) const {
  if (!node) return nullptr;

  for (const auto& child : node->children()) {
    if (child->value() == value &&
        child->level() == level &&
        child->isShape() == isShape &&
        (shapeHash == 0 || child->shapeHash() == shapeHash)) {
      return child;
    }
  }
  return nullptr;
}

ConstantShapeBuffer* DirectShapeTrie::search(const LongType* shapeInfo, size_t stripeIdx) const {
  // No locks here - caller handles locking
  const ShapeTrieNode* current = _roots[stripeIdx];
  const int rank = shape::rank(shapeInfo);
  const int shapeSignature = calculateShapeSignature(shapeInfo);

  // Check rank
  current = findChild(current, rank, 0, true, shapeSignature);
  if (!current) return nullptr;

  // Check datatype
  current = findChild(current, ArrayOptions::dataType(shapeInfo), 1, true, shapeSignature);
  if (!current) return nullptr;

  // Check order
  current = findChild(current, shape::order(shapeInfo), 2, true, shapeSignature);
  if (!current) return nullptr;

  // Check shape values
  const LongType* shapeValues = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = findChild(current, shapeValues[i], 3 + i, true, shapeSignature);
    if (!current) return nullptr;
  }

  // Check stride values
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = findChild(current, strides[i], 3 + rank + i, false, shapeSignature);
    if (!current) return nullptr;
  }

  return current ? current->buffer() : nullptr;
}

// Updated getOrCreate method with improved thread safety
ConstantShapeBuffer* DirectShapeTrie::getOrCreate(const LongType* shapeInfo) {
  validateShapeInfo(shapeInfo);
  DataType inputType = ArrayOptions::dataType(shapeInfo);
  printf("DirectShapeTrie::getOrCreate input type: %d\n", (int)inputType);

  size_t stripeIdx = getStripeIndex(shapeInfo);
  int shapeSignature = calculateShapeSignature(shapeInfo);

  // First try a read-only lookup without obtaining a write lock
  {
    std::shared_lock<SHAPE_MUTEX_TYPE> readLock(_mutexes[stripeIdx]);
    ConstantShapeBuffer* existing = search(shapeInfo, stripeIdx);
    if (existing != nullptr) {
      // Verify that the shapes match exactly
      if (shapeInfoEqual(existing->primary(), shapeInfo)) {
        return existing;
      }
    }
  }

  // If not found or not matching, grab exclusive lock and try again
  std::unique_lock<SHAPE_MUTEX_TYPE> writeLock(_mutexes[stripeIdx]);

  // Check again under the write lock
  ConstantShapeBuffer* existing = search(shapeInfo, stripeIdx);
  if (existing != nullptr) {
    // Double-check the shape match
    if (shapeInfoEqual(existing->primary(), shapeInfo)) {
      return existing;
    }
  }

  // Not found or not matching, need to create a new shape buffer
  ShapeTrieNode* current = _roots[stripeIdx];
  const int rank = shape::rank(shapeInfo);

  // Insert rank with signature
  current = current->findOrCreateChild(rank, 0, true, shapeSignature);
  if (rank < 0 || rank > SD_MAX_RANK) {
    printf("ERROR: Invalid rank %lld in DirectShapeTrie::getOrCreate\n", (long long)rank);
    THROW_EXCEPTION("Invalid rank in shape trie");
  }

  if (!current) {
    THROW_EXCEPTION("Failed to create rank node");
  }

  // Insert datatype with signature
  current = current->findOrCreateChild(ArrayOptions::dataType(shapeInfo), 1, true, shapeSignature);
  if (!current) {
    THROW_EXCEPTION("Failed to create datatype node");
  }

  // Insert order with signature
  current = current->findOrCreateChild(shape::order(shapeInfo), 2, true, shapeSignature);
  if (!current) {
    THROW_EXCEPTION("Failed to create order node");
  }

  // Insert shape values with signature
  const LongType* shapeValues = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(shapeValues[i], 3 + i, true, shapeSignature);
    if (!current) {
      THROW_EXCEPTION("Failed to create shape value node");
    }
  }

  // Insert stride values with signature
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(strides[i], 3 + rank + i, false, shapeSignature);
    if (!current) {
      THROW_EXCEPTION("Failed to create stride value node");
    }
  }

  // Check if another thread has already created the buffer
  if (ConstantShapeBuffer* nodeExisting = current->buffer()) {
    if (shapeInfoEqual(nodeExisting->primary(), shapeInfo)) {
      return nodeExisting;
    }
  }


  // Create the shape buffer
  ConstantShapeBuffer* buffer = ShapeBufferCreatorHelper::getCurrentCreator().create(shapeInfo, rank);
  DataType resultType = ArrayOptions::dataType(buffer->primary());
  if (inputType != resultType) {
    printf("ERROR: Data type changed from %d to %d in DirectShapeTrie::getOrCreate\n",
           (int)inputType, (int)resultType);
  }

  current->setBuffer(buffer);
  return buffer;

  return nullptr;
}

bool DirectShapeTrie::exists(const LongType* shapeInfo) const {
  validateShapeInfo(shapeInfo);
  size_t stripeIdx = getStripeIndex(shapeInfo);
  int shapeSignature = calculateShapeSignature(shapeInfo);

  std::shared_lock<SHAPE_MUTEX_TYPE> lock(_mutexes[stripeIdx]);
  ConstantShapeBuffer* found = search(shapeInfo, stripeIdx);
  return found != nullptr && shapeInfoEqual(found->primary(), shapeInfo);
}

// Original insert method kept for compatibility, but getOrCreate should be used instead
ConstantShapeBuffer* DirectShapeTrie::insert(const LongType* shapeInfo, size_t stripeIdx) {
  ShapeTrieNode* current = _roots[stripeIdx];
  const int rank = shape::rank(shapeInfo);
  const int shapeSignature = calculateShapeSignature(shapeInfo);

  // Insert rank
  current = current->findOrCreateChild(rank, 0, true, shapeSignature);
  if (!current) return nullptr;

  // Insert datatype
  current = current->findOrCreateChild(ArrayOptions::dataType(shapeInfo), 1, true, shapeSignature);
  if (!current) return nullptr;

  // Insert order
  current = current->findOrCreateChild(shape::order(shapeInfo), 2, true, shapeSignature);
  if (!current) return nullptr;

  // Insert shape values
  const LongType* shape = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(shape[i], 3 + i, true, shapeSignature);
    if (!current) return nullptr;
  }

  // Insert stride values
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(strides[i], 3 + rank + i, false, shapeSignature);
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

}  // namespace sd