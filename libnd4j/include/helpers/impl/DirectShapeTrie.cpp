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
  if (!buf) return;  // Nothing to do if buffer is null

  // If we already have a buffer, don't replace it
  if (_buffer != nullptr) {
    // The existing buffer takes precedence
    // Only delete the new buffer if it's different and not needed elsewhere
    if (buf != _buffer) {
      delete buf;  // This buffer is redundant, we already have one
    }
    return;
  }

  // At this point, we know _buffer is null and buf is valid
  // Set the buffer atomically
  _buffer = buf;
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
    std::string msg = "Shape info cannot be null";
    THROW_EXCEPTION(msg.c_str());
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
      std::string msg = "Found shape buffer with all zero values. Values likely unset.";
      THROW_EXCEPTION(msg.c_str());
    }
  }

  if (ArrayOptions::dataType(shapeInfo) == UNKNOWN) {
    std::string msg = "Shape info created with invalid data type";
    THROW_EXCEPTION(msg.c_str());
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

// Modified search method - still returns null when shape not found but with improved debugging
ConstantShapeBuffer* DirectShapeTrie::search(const LongType* shapeInfo, size_t stripeIdx) const {
  // Validate input
  if (shapeInfo == nullptr) {
    std::string msg = "Null shapeInfo passed to search method";
    THROW_EXCEPTION(msg.c_str());
  }

  if (stripeIdx >= NUM_STRIPES) {
    std::string msg = "Invalid stripe index: " + std::to_string(stripeIdx) +
                      " (max: " + std::to_string(NUM_STRIPES - 1) + ")";
    THROW_EXCEPTION(msg.c_str());
  }

  if (_roots == nullptr) {
    std::string msg = "Root nodes array is null";
    THROW_EXCEPTION(msg.c_str());
  }
  auto rootsRef = *_roots;


  // No locks here - caller handles locking
  const ShapeTrieNode* current = rootsRef[stripeIdx];
  if (current == nullptr) {
    // Cannot use createFallbackBuffer here as it's const method
    // Caller should handle this case
    return nullptr;
  }

  const int rank = shape::rank(shapeInfo);
  const int shapeSignature = calculateShapeSignature(shapeInfo);

  // Check rank
  current = findChild(current, rank, 0, true, shapeSignature);
  if (!current) {
    return nullptr;  // Not found, but this is expected behavior
  }

  // Check datatype
  current = findChild(current, ArrayOptions::dataType(shapeInfo), 1, true, shapeSignature);
  if (!current) {
    return nullptr;  // Not found, but this is expected behavior
  }

  // Check order
  current = findChild(current, shape::order(shapeInfo), 2, true, shapeSignature);
  if (!current) {
    return nullptr;  // Not found, but this is expected behavior
  }

  // Check shape values
  const LongType* shapeValues = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = findChild(current, shapeValues[i], 3 + i, true, shapeSignature);
    if (!current) {
      return nullptr;  // Not found, but this is expected behavior
    }
  }

  // Check stride values
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = findChild(current, strides[i], 3 + rank + i, false, shapeSignature);
    if (!current) {
      return nullptr;  // Not found, but this is expected behavior
    }
  }

  return current ? current->buffer() : nullptr;
}


// Helper method to create a fallback buffer when the trie insertion fails
ConstantShapeBuffer* DirectShapeTrie::createFallbackBuffer(const LongType* shapeInfo, int rank) {
  if (shapeInfo == nullptr) {
    std::string msg = "Null shapeInfo passed to createFallbackBuffer";
    THROW_EXCEPTION(msg.c_str());
  }

  if (rank < 0 || rank > SD_MAX_RANK) {
    std::string msg = "Invalid rank in createFallbackBuffer: " + std::to_string(rank);
    THROW_EXCEPTION(msg.c_str());
  }

  // Create a direct copy of the shape info
  const int shapeInfoLength = shape::shapeInfoLength(rank);
  LongType* shapeCopy = new LongType[shapeInfoLength];
  if (shapeCopy == nullptr) {
    std::string msg = "Failed to allocate memory for shape copy";
    THROW_EXCEPTION(msg.c_str());
  }

  std::memcpy(shapeCopy, shapeInfo, shapeInfoLength * sizeof(LongType));

  // Create a deallocator for memory management
  auto deallocator = std::shared_ptr<PrimaryPointerDeallocator>(
      new PrimaryPointerDeallocator(),
      [] (PrimaryPointerDeallocator* ptr) { delete ptr; });

  // Create a pointer wrapper and buffer
  auto hPtr = new PointerWrapper(shapeCopy, deallocator);
  if (hPtr == nullptr) {
    delete[] shapeCopy;
    std::string msg = "Failed to create PointerWrapper";
    THROW_EXCEPTION(msg.c_str());
  }

  auto buffer = new ConstantShapeBuffer(hPtr);
  if (buffer == nullptr) {
    delete hPtr;
    std::string msg = "Failed to create ConstantShapeBuffer";
    THROW_EXCEPTION(msg.c_str());
  }

  return buffer;
}

// Updated getOrCreate method to ensure it always creates a shape buffer
ConstantShapeBuffer* DirectShapeTrie::getOrCreate(const LongType* shapeInfo) {
  if (!shapeInfo) {
    std::string msg = "Null shapeInfo passed to getOrCreate";
    THROW_EXCEPTION(msg.c_str());
  }

  validateShapeInfo(shapeInfo);

  size_t stripeIdx = getStripeIndex(shapeInfo);
  int rank = shape::rank(shapeInfo);

  // Validate stripe index
  if (stripeIdx >= NUM_STRIPES) {
    stripeIdx = NUM_STRIPES - 1;
  }

  int shapeSignature = calculateShapeSignature(shapeInfo);

  // Check if mutex pointer is valid
  if (_mutexes == nullptr || (*_mutexes)[stripeIdx] == nullptr) {
    return createFallbackBuffer(shapeInfo, rank);
  }

  // First try a read-only lookup without obtaining a write lock
  {
   LOCK_TYPE<MUTEX_TYPE> readLock(*(*_mutexes)[stripeIdx]);
    ConstantShapeBuffer* existing = search(shapeInfo, stripeIdx);
    if (existing != nullptr) {
      if (shapeInfoEqual(existing->primary(), shapeInfo)) {
        return existing;
      }
    }
  }

  // If not found or not matching, grab exclusive lock and try again
  LOCK_TYPE<MUTEX_TYPE> writeLock(*(*_mutexes)[stripeIdx]);

  // Check again under the write lock
  ConstantShapeBuffer* existing = search(shapeInfo, stripeIdx);
  if (existing != nullptr) {
    if (shapeInfoEqual(existing->primary(), shapeInfo)) {
      return existing;
    }
  }

  if (_roots == nullptr) {
    return createFallbackBuffer(shapeInfo, rank);
  }
  // Not found, create a new entry
  auto rootsRef = *_roots;


  ShapeTrieNode* current = rootsRef[stripeIdx];
  if (current == nullptr) {
    return createFallbackBuffer(shapeInfo, rank);
  }

  if (rank < 0 || rank > SD_MAX_RANK) {
    return createFallbackBuffer(shapeInfo, rank);
  }

  // Safe pointer to track the current node through the insertion process
  ShapeTrieNode* safeNodePtr = nullptr;

  // Insert rank with signature
  safeNodePtr = current->findOrCreateChild(rank, 0, true, shapeSignature);
  if (safeNodePtr == nullptr) {
    return createFallbackBuffer(shapeInfo, rank);
  }
  current = safeNodePtr;

  // Insert datatype with signature
  safeNodePtr = current->findOrCreateChild(ArrayOptions::dataType(shapeInfo), 1, true, shapeSignature);
  if (safeNodePtr == nullptr) {
    return createFallbackBuffer(shapeInfo, rank);
  }
  current = safeNodePtr;

  // Insert order with signature
  safeNodePtr = current->findOrCreateChild(shape::order(shapeInfo), 2, true, shapeSignature);
  if (safeNodePtr == nullptr) {
    return createFallbackBuffer(shapeInfo, rank);
  }
  current = safeNodePtr;

  // Insert shape values with signature
  const LongType* shapeValues = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    safeNodePtr = current->findOrCreateChild(shapeValues[i], 3 + i, true, shapeSignature);
    if (safeNodePtr == nullptr) {
      return createFallbackBuffer(shapeInfo, rank);
    }
    current = safeNodePtr;
  }

  // Insert stride values with signature
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    safeNodePtr = current->findOrCreateChild(strides[i], 3 + rank + i, false, shapeSignature);
    if (safeNodePtr == nullptr) {
      return createFallbackBuffer(shapeInfo, rank);
    }
    current = safeNodePtr;
  }

  // Check if another thread has already created the buffer
  if (ConstantShapeBuffer* nodeBuffer = current->buffer()) {
    if (shapeInfoEqual(nodeBuffer->primary(), shapeInfo)) {
      return nodeBuffer;
    }
  }

  // Create the shape buffer
  ConstantShapeBuffer* buffer = ShapeBufferCreatorHelper::getCurrentCreator().create(shapeInfo, rank);
  if (buffer == nullptr || buffer->primary() == nullptr) {
    // Use fallback if creator fails
    if (buffer != nullptr) {
      delete buffer;  // Clean up invalid buffer
    }
    return createFallbackBuffer(shapeInfo, rank);
  }

  // Set the buffer - setBuffer handles ownership properly
  current->setBuffer(buffer);

  // Return the buffer from the node (could be the one we just set or a pre-existing one)
  ConstantShapeBuffer* resultBuffer = current->buffer();
  if (resultBuffer == nullptr) {
    return buffer;
  }

  return resultBuffer;
}

bool DirectShapeTrie::exists(const LongType* shapeInfo) const {
  validateShapeInfo(shapeInfo);
  size_t stripeIdx = getStripeIndex(shapeInfo);

  // Validate stripe index
  if (stripeIdx >= NUM_STRIPES) {
    return false;
  }

  // Check if mutex pointer is valid
  if (_mutexes == nullptr || (*_mutexes)[stripeIdx] == nullptr) {
    return false;
  }

  int shapeSignature = calculateShapeSignature(shapeInfo);

  LOCK_TYPE<MUTEX_TYPE> lock(*(*_mutexes)[stripeIdx]);
  ConstantShapeBuffer* found = search(shapeInfo, stripeIdx);
  return found != nullptr && shapeInfoEqual(found->primary(), shapeInfo);
}

// Original insert method kept for compatibility, but getOrCreate should be used instead
ConstantShapeBuffer* DirectShapeTrie::insert(const LongType* shapeInfo, size_t stripeIdx) {
  auto rootsRef = *_roots;

  ShapeTrieNode* current = rootsRef[stripeIdx];
  const int rank = shape::rank(shapeInfo);
  const int shapeSignature = calculateShapeSignature(shapeInfo);

  // Insert rank
  current = current->findOrCreateChild(rank, 0, true, shapeSignature);
  if (!current) {
    std::string msg = "Failed to create rank node";
    THROW_EXCEPTION(msg.c_str());
    return nullptr;
  }

  // Insert datatype
  current = current->findOrCreateChild(ArrayOptions::dataType(shapeInfo), 1, true, shapeSignature);
  if (!current) {
    std::string msg = "Failed to create datatype node";
    THROW_EXCEPTION(msg.c_str());
    return nullptr;
  }

  // Insert order
  current = current->findOrCreateChild(shape::order(shapeInfo), 2, true, shapeSignature);
  if (!current) {
    std::string msg = "Failed to create order node";
    THROW_EXCEPTION(msg.c_str());
    return nullptr;
  }

  // Insert shape values
  const LongType* shape = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(shape[i], 3 + i, true, shapeSignature);
    if (!current) {
      std::string msg = "Failed to create shape value node at index " + std::to_string(i);
      THROW_EXCEPTION(msg.c_str());
      return nullptr;
    }
  }

  // Insert stride values
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(strides[i], 3 + rank + i, false, shapeSignature);
    if (!current) {
      std::string msg = "Failed to create stride value node at index " + std::to_string(i);
      THROW_EXCEPTION(msg.c_str());
      return nullptr;
    }
  }

  if (!current->buffer()) {
    try {
      const int shapeInfoLength = shape::shapeInfoLength(rank);
      LongType* shapeCopy = new LongType[shapeInfoLength];
      std::memcpy(shapeCopy, shapeInfo, shapeInfoLength * sizeof(LongType));

      auto deallocator = std::shared_ptr<PrimaryPointerDeallocator>(new PrimaryPointerDeallocator(),
                                                                    [] (PrimaryPointerDeallocator* ptr) { delete ptr; });
      auto hPtr = new PointerWrapper(shapeCopy, deallocator);
      auto buffer = new ConstantShapeBuffer(hPtr);

      current->setBuffer(buffer);
      return buffer;
    } catch (const std::exception& e) {
      std::string msg = "Shape buffer creation failed: ";
      msg += e.what();
      THROW_EXCEPTION(msg.c_str());
    } catch (...) {
      std::string msg = "Shape buffer creation failed with unknown exception";
      THROW_EXCEPTION(msg.c_str());
    }
  }

  return current->buffer();
}

}  // namespace sd