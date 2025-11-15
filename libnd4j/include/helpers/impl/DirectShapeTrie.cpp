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

#include <array/ArrayOptions.hXX>
#include <array/ConstantShapeBuffer.h>
#include <array/DataType.h>
#include <array/PrimaryPointerDeallocator.h>
#include <helpers/DirectShapeTrie.h>
#include <helpers/shape.h>
#include <system/common.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <string>

#include "helpers/ShapeBufferCreatorHelper.h"

#if defined(SD_GCC_FUNCTRACE)
#include <array/ShapeCacheLifecycleTracker.h>
#endif

namespace sd {

void ShapeTrieNode::setBuffer(ConstantShapeBuffer* buf) {
  if (!buf) return;  // Nothing to do if buffer is null

  // If we already have a buffer, don't replace it
  if (_buffer != nullptr) {
    // The existing buffer takes precedence
    // Don't delete the new buffer - let the caller handle it
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

  // **NEW: Add property flags to distinguish views from non-views**
  hash = hash * 47 + static_cast<size_t>(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);

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

  // **NEW: Include property flags**
  signature = signature * 29 + static_cast<int>(shapeInfo[ArrayOptions::extraIndex(shapeInfo)] % 10000);

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

#if defined(SD_GCC_FUNCTRACE)
  // Track shape cache allocation
  sd::array::ShapeCacheLifecycleTracker::getInstance().recordAllocation(shapeCopy);
#endif

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
    SHARED_LOCK_TYPE<MUTEX_TYPE> readLock(*(*_mutexes)[stripeIdx]);
    ConstantShapeBuffer* existing = search(shapeInfo, stripeIdx);
    if (existing != nullptr) {
      if (shapeInfoEqual(existing->primary(), shapeInfo)) {
        return existing;
      }
    }
  }

  // If not found or not matching, grab exclusive lock and try again
  SHARED_LOCK_TYPE<MUTEX_TYPE> writeLock(*(*_mutexes)[stripeIdx]);

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

  SHARED_LOCK_TYPE<MUTEX_TYPE> lock(*(*_mutexes)[stripeIdx]);
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

#if defined(SD_GCC_FUNCTRACE)
      // Track shape cache allocation
      sd::array::ShapeCacheLifecycleTracker::getInstance().recordAllocation(shapeCopy);
#endif

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

void DirectShapeTrie::clearCache() {
  if (_roots == nullptr || _mutexes == nullptr) {
    return;
  }

  // Clear each stripe
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    MUTEX_TYPE* mutex = (*_mutexes)[i];
    if (mutex == nullptr) continue;

    // Lock this stripe
    std::lock_guard<MUTEX_TYPE> lock(*mutex);

    // Delete the old root node (destructor recursively cleans up all children and buffers)
    ShapeTrieNode* oldRoot = (*_roots)[i];
    if (oldRoot != nullptr) {
      delete oldRoot;
    }

    // Create a new empty root node
    (*_roots)[i] = new ShapeTrieNode(0, 0, false);
  }

  // Reset current counters (but preserve peak values for diagnostics)
  _current_entries.store(0);
  _current_bytes.store(0);
}

void DirectShapeTrie::countEntriesAndBytes(const ShapeTrieNode* node, LongType& entries, LongType& bytes) const {
  if (node == nullptr) return;

  // If this node has a buffer, count it
  ConstantShapeBuffer* buffer = node->buffer();
  if (buffer != nullptr) {
    entries++;
    // Calculate buffer size: shapeInfo length is stored at index 0
    const LongType* shapeInfo = buffer->primary();
    if (shapeInfo != nullptr) {
      LongType bufferLength = shape::shapeInfoLength(shapeInfo);
      bytes += bufferLength * sizeof(LongType);
    }
  }

  // Recursively count children
  const std::vector<ShapeTrieNode*>& children = node->children();
  for (const auto* child : children) {
    countEntriesAndBytes(child, entries, bytes);
  }
}

LongType DirectShapeTrie::getCachedEntries() const {
  LongType total_entries = 0;
  LongType total_bytes = 0;

  if (_roots == nullptr || _mutexes == nullptr) {
    return 0;
  }

  // Count entries across all stripes
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    MUTEX_TYPE* mutex = (*_mutexes)[i];
    if (mutex == nullptr) continue;

    // Lock this stripe for reading
    std::lock_guard<MUTEX_TYPE> lock(*mutex);

    ShapeTrieNode* root = (*_roots)[i];
    if (root != nullptr) {
      countEntriesAndBytes(root, total_entries, total_bytes);
    }
  }

  // Update current counters
  _current_entries.store(total_entries);
  _current_bytes.store(total_bytes);

  // Update peak if current exceeds it
  LongType current_peak = _peak_entries.load();
  while (total_entries > current_peak) {
    if (_peak_entries.compare_exchange_weak(current_peak, total_entries)) {
      break;
    }
  }

  current_peak = _peak_bytes.load();
  while (total_bytes > current_peak) {
    if (_peak_bytes.compare_exchange_weak(current_peak, total_bytes)) {
      break;
    }
  }

  return total_entries;
}

LongType DirectShapeTrie::getCachedBytes() const {
  // getCachedEntries() updates both entries and bytes
  getCachedEntries();
  return _current_bytes.load();
}

LongType DirectShapeTrie::getPeakCachedEntries() const {
  return _peak_entries.load();
}

LongType DirectShapeTrie::getPeakCachedBytes() const {
  return _peak_bytes.load();
}

void DirectShapeTrie::buildStringRepresentation(const ShapeTrieNode* node, std::stringstream& ss,
                                                const std::string& indent, int currentDepth,
                                                int maxDepth, int& entriesShown, int maxEntries) const {
  if (node == nullptr) return;
  if (maxDepth != -1 && currentDepth > maxDepth) return;
  if (maxEntries != -1 && entriesShown >= maxEntries) return;

  // Check if this node has a buffer
  ConstantShapeBuffer* buffer = node->buffer();
  if (buffer != nullptr) {
    const LongType* shapeInfo = buffer->primary();
    if (shapeInfo != nullptr) {
      entriesShown++;

      // Display node info
      ss << indent << "Node[level=" << node->level()
         << ", value=" << node->value()
         << ", isShape=" << (node->isShape() ? "true" : "false")
         << "]\n";

      // Display shape info details
      int rank = shape::rank(shapeInfo);
      ss << indent << "  Shape: rank=" << rank << ", order=" << shape::order(shapeInfo)
         << ", dtype=" << DataTypeUtils::asString(ArrayOptions::dataType(shapeInfo)) << "\n";

      // Display shape dimensions
      ss << indent << "  Dims: [";
      const LongType* dims = shape::shapeOf(shapeInfo);
      for (int i = 0; i < rank; i++) {
        if (i > 0) ss << ", ";
        ss << dims[i];
      }
      ss << "]\n";

      // Display strides
      ss << indent << "  Strides: [";
      const LongType* strides = shape::stride(shapeInfo);
      for (int i = 0; i < rank; i++) {
        if (i > 0) ss << ", ";
        ss << strides[i];
      }
      ss << "]\n";

      // Display total elements and buffer size
      LongType length = shape::length(shapeInfo);
      LongType bufferLength = shape::shapeInfoLength(shapeInfo);
      ss << indent << "  Elements: " << length
         << ", Buffer size: " << (bufferLength * sizeof(LongType)) << " bytes\n";

      if (maxEntries != -1 && entriesShown >= maxEntries) {
        ss << indent << "  ... (max entries reached)\n";
        return;
      }
    }
  }

  // Recursively process children
  const std::vector<ShapeTrieNode*>& children = node->children();
  if (!children.empty() && (maxDepth == -1 || currentDepth < maxDepth)) {
    for (const auto* child : children) {
      if (maxEntries != -1 && entriesShown >= maxEntries) break;
      buildStringRepresentation(child, ss, indent + "  ", currentDepth + 1,
                               maxDepth, entriesShown, maxEntries);
    }
  }
}

std::string DirectShapeTrie::toString(int maxDepth, int maxEntries) const {
  std::stringstream ss;

  if (_roots == nullptr || _mutexes == nullptr) {
    ss << "DirectShapeTrie: [UNINITIALIZED]\n";
    return ss.str();
  }

  // Get current statistics
  LongType totalEntries = getCachedEntries();
  LongType totalBytes = getCachedBytes();
  LongType peakEntries = getPeakCachedEntries();
  LongType peakBytes = getPeakCachedBytes();

  // Header
  ss << "DirectShapeTrie [" << NUM_STRIPES << " stripes]\n";
  ss << "Current: " << totalEntries << " entries, " << totalBytes << " bytes\n";
  ss << "Peak: " << peakEntries << " entries, " << peakBytes << " bytes\n";
  ss << "Showing: max depth=" << (maxDepth == -1 ? "unlimited" : std::to_string(maxDepth))
     << ", max entries=" << (maxEntries == -1 ? "unlimited" : std::to_string(maxEntries)) << "\n";
  ss << "---\n";

  int entriesShown = 0;

  // Traverse each stripe
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    MUTEX_TYPE* mutex = (*_mutexes)[i];
    if (mutex == nullptr) continue;

    // Lock this stripe for reading
    std::lock_guard<MUTEX_TYPE> lock(*mutex);

    ShapeTrieNode* root = (*_roots)[i];
    if (root != nullptr && !root->children().empty()) {
      ss << "Stripe " << i << ":\n";
      buildStringRepresentation(root, ss, "  ", 0, maxDepth, entriesShown, maxEntries);

      if (maxEntries != -1 && entriesShown >= maxEntries) {
        ss << "... (max entries limit reached, " << (totalEntries - entriesShown)
           << " more entries not shown)\n";
        break;
      }
    }
  }

  if (entriesShown == 0) {
    ss << "(Cache is empty)\n";
  }

  return ss.str();
}

void DirectShapeTrie::getCachedPointers(std::unordered_set<void*>& out_pointers) const {
  if (_roots == nullptr || _mutexes == nullptr) {
    return;
  }

  // Traverse all stripes and collect ConstantShapeBuffer pointers
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    MUTEX_TYPE* mutex = (*_mutexes)[i];
    if (mutex == nullptr) continue;

    std::lock_guard<MUTEX_TYPE> lock(*mutex);

    ShapeTrieNode* root = (*_roots)[i];
    if (root != nullptr) {
      collectCachedPointers(root, out_pointers);
    }
  }
}

void DirectShapeTrie::collectCachedPointers(const ShapeTrieNode* node, std::unordered_set<void*>& out_pointers) const {
  if (node == nullptr) return;

  // If this node has a ConstantShapeBuffer, add it to the set
  ConstantShapeBuffer* buffer = node->buffer();
  if (buffer != nullptr) {
    out_pointers.insert(buffer);
  }

  // Recursively collect from all children
  for (const auto* child : node->children()) {
    collectCachedPointers(child, out_pointers);
  }
}

}  // namespace sd