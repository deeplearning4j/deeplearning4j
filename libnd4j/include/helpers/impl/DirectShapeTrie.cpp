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
#include <execution/AffinityManager.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "helpers/ShapeBufferCreatorHelper.h"

#if defined(SD_GCC_FUNCTRACE)
#include <array/ShapeCacheLifecycleTracker.h>
#endif

namespace sd {

void DirectShapeTrie::waitForInitialization() const {
  if (_initialization_complete.load(std::memory_order_acquire)) {
    return;
  }

  int attempts = 0;
  while (_initialization_in_progress.load(std::memory_order_acquire)) {
    if (attempts < 10) {
      std::this_thread::yield();
    } else {
      auto delay = std::chrono::microseconds(std::min<int>(500, attempts * 10));
      std::this_thread::sleep_for(delay);
    }
    attempts++;
  }

  if (!_initialization_complete.load(std::memory_order_acquire)) {
    THROW_EXCEPTION("DirectShapeTrie initialization did not complete before use");
  }
}

void ShapeTrieNode::setBuffer(ConstantShapeBuffer* buf) {
  if (!buf) return;  // Nothing to do if buffer is null

  ConstantShapeBuffer* expected = nullptr;
  if (_buffer.compare_exchange_strong(expected, buf, std::memory_order_acq_rel)) {
    // Successfully set the buffer - we are the first thread
    // The buffer is now owned by the cache, addRef() was already done by caller
    return;
  }

  // Another thread already set a buffer (expected now contains the existing buffer).
  // Don't replace it - the first buffer wins to maintain consistency.
  // The caller must handle the unused buffer (delete it or return it).
}


#if defined(SD_GCC_FUNCTRACE)
void ShapeTrieNode::collectStoreStackTrace() {
  this->storeStackTrace = backward::StackTrace();
  this->storeStackTrace.load_here(32);
}
#endif

size_t DirectShapeTrie::computeHash(const LongType* shapeInfo) const {
  size_t hash = 17;
  hash = hash * 53 + static_cast<size_t>(AffinityManager::currentDeviceId()) * 59;
  const int descriptorLength =
      static_cast<int>(shape::shapeInfoLength(shape::rank(shapeInfo)));
  hash = hash * 31 + static_cast<size_t>(descriptorLength);
  // Hash every stored descriptor word. Rank-0 descriptors retain scalar
  // shape/stride storage and a raw order word even though their logical shape
  // has no dimensions. Logical accessors omit those words and therefore cannot
  // provide a collision-free cache identity.
  for (int i = 0; i < descriptorLength; ++i) {
    hash = hash * 131 + static_cast<size_t>(shapeInfo[i]);
  }
  return hash;
}

int DirectShapeTrie::calculateShapeSignature(const LongType* shapeInfo) const {
  uint32_t signature = 2166136261u;
  signature = (signature ^ static_cast<uint32_t>(AffinityManager::currentDeviceId())) * 16777619u;
  const int descriptorLength =
      static_cast<int>(shape::shapeInfoLength(shape::rank(shapeInfo)));
  for (int i = 0; i < descriptorLength; ++i) {
    const uint64_t word = static_cast<uint64_t>(shapeInfo[i]);
    signature = (signature ^ static_cast<uint32_t>(word)) * 16777619u;
    signature = (signature ^ static_cast<uint32_t>(word >> 32)) * 16777619u;
  }
  // Zero is the findChild wildcard, so reserve it rather than weakening lookup.
  return static_cast<int>(signature == 0 ? 1u : signature);
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

  const int shapeSignature = calculateShapeSignature(shapeInfo);

  current = findChild(current, AffinityManager::currentDeviceId(), 0, false,
                      shapeSignature);
  if (!current) {
    return nullptr;  // Not found for this device
  }
  const int descriptorLength =
      static_cast<int>(shape::shapeInfoLength(shape::rank(shapeInfo)));
  for (int i = 0; i < descriptorLength; ++i) {
    current = findChild(current, shapeInfo[i], 1 + i, false, shapeSignature);
    if (!current) return nullptr;
  }

  return current->buffer();
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

  // Use platform-specific creator (CudaShapeBufferCreator for CUDA, CpuShapeBufferCreator for CPU)
  // Note: The creator makes its own copy of shapeInfo internally, so we don't need to copy here.
  auto buffer = ShapeBufferCreatorHelper::getCurrentCreator().create(shapeInfo, rank);
  if (buffer == nullptr || buffer->primary() == nullptr) {
    std::string msg = "Failed to create ConstantShapeBuffer via platform creator";
    THROW_EXCEPTION(msg.c_str());
  }

#if defined(SD_GCC_FUNCTRACE)
  // Track shape cache allocation - use the buffer's primary pointer (the creator's copy)
  sd::array::ShapeCacheLifecycleTracker::getInstance().recordAllocation(buffer->primary());
#endif

  // Fallback buffer is NOT cached, so refCount stays at 1 (caller owns it)
  // Caller will call deleteConstantShapeBuffer() which calls release()
  return buffer;
}

// Updated getOrCreate method to ensure it always creates a shape buffer
ConstantShapeBuffer* DirectShapeTrie::getOrCreate(const LongType* shapeInfo) {
  waitForInitialization();

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
      if (!shapeInfoEqual(existing->primary(), shapeInfo)) {
        THROW_EXCEPTION("DirectShapeTrie: read lookup returned a shape buffer whose full descriptor does not match the key");
      }
      existing->addRef();  // Increment refcount before returning cached buffer
      return existing;
    }
  }

  // If not found, grab exclusive lock and try again
  EXCLUSIVE_LOCK_TYPE<MUTEX_TYPE> writeLock(*(*_mutexes)[stripeIdx]);

  // Check again under the write lock
  ConstantShapeBuffer* existing = search(shapeInfo, stripeIdx);
  if (existing != nullptr) {
    if (!shapeInfoEqual(existing->primary(), shapeInfo)) {
      THROW_EXCEPTION("DirectShapeTrie: write lookup returned a shape buffer whose full descriptor does not match the key");
    }
    existing->addRef();  // Increment refcount before returning cached buffer
    return existing;
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

  safeNodePtr = current->findOrCreateChild(
      AffinityManager::currentDeviceId(), 0, false, shapeSignature);
  if (safeNodePtr == nullptr) {
    return createFallbackBuffer(shapeInfo, rank);
  }
  current = safeNodePtr;
  const int descriptorLength =
      static_cast<int>(shape::shapeInfoLength(rank));
  for (int i = 0; i < descriptorLength; ++i) {
    safeNodePtr = current->findOrCreateChild(
        shapeInfo[i], 1 + i, false, shapeSignature);
    if (safeNodePtr == nullptr) return createFallbackBuffer(shapeInfo, rank);
    current = safeNodePtr;
  }

  // Check if another thread has already created the buffer
  if (ConstantShapeBuffer* nodeBuffer = current->buffer()) {
    if (shapeInfoEqual(nodeBuffer->primary(), shapeInfo)) {
      nodeBuffer->addRef();  // Increment refcount before returning cached buffer
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

  // Try to set the buffer atomically - setBuffer uses compare-and-swap.
  // Under the exclusive write lock, no other thread can be in this section
  // for the same stripe, so the CAS should always succeed (node was just
  // created by findOrCreateChild above with _buffer == nullptr).
  current->setBuffer(buffer);

  // Read back the buffer from the node.
  ConstantShapeBuffer* resultBuffer = current->buffer();

  if (resultBuffer == buffer) {
    // We won the CAS — buffer is now stored in the trie (implicit ref via
    // the initial refCount=1 from construction). Bump refCount for the
    // caller so the buffer survives even if the trie is cleared.
    buffer->addRef();
    return buffer;
  }

  // CAS failed — another buffer was already there (shouldn't happen under
  // the exclusive lock, but handle defensively). Delete our unused buffer
  // and return the existing one with a caller ref.
  delete buffer;

  if (resultBuffer != nullptr) {
    resultBuffer->addRef();
  }
  return resultBuffer;
}

bool DirectShapeTrie::exists(const LongType* shapeInfo) const {
  waitForInitialization();

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

  current = current->findOrCreateChild(
      AffinityManager::currentDeviceId(), 0, false, shapeSignature);
  if (!current) {
    std::string msg = "Failed to create device node";
    THROW_EXCEPTION(msg.c_str());
    return nullptr;
  }
  const int descriptorLength =
      static_cast<int>(shape::shapeInfoLength(rank));
  for (int i = 0; i < descriptorLength; ++i) {
    current = current->findOrCreateChild(
        shapeInfo[i], 1 + i, false, shapeSignature);
    if (!current) {
      std::string msg = "Failed to create descriptor node at index " +
                        std::to_string(i);
      THROW_EXCEPTION(msg.c_str());
      return nullptr;
    }
  }

  if (!current->buffer()) {
    try {
      // Use platform-specific creator (CudaShapeBufferCreator for CUDA, CpuShapeBufferCreator for CPU)
      auto buffer = ShapeBufferCreatorHelper::getCurrentCreator().create(shapeInfo, rank);
      if (buffer == nullptr || buffer->primary() == nullptr) {
        std::string msg = "Failed to create ConstantShapeBuffer via platform creator in search";
        THROW_EXCEPTION(msg.c_str());
      }

#if defined(SD_GCC_FUNCTRACE)
      // Track shape cache allocation
      sd::array::ShapeCacheLifecycleTracker::getInstance().recordAllocation(buffer->primary());
#endif

      current->setBuffer(buffer);
      // Buffer is now cached (trie owns it with refCount=1)
      // Increment refcount so caller also has a reference
      buffer->addRef();
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

  ConstantShapeBuffer* result = current->buffer();
  if (result != nullptr) {
    result->addRef();  // Increment refcount before returning cached buffer
  }
  return result;
}

void DirectShapeTrie::clearCache() {
  // Check shutdown flag first - if set, let the OS reclaim memory at exit
  // This prevents segfaults during JVM shutdown when buffers may still have
  // external references or memory allocators may be in an inconsistent state
  if (_shutdownInProgress.load(std::memory_order_acquire)) {
    return;
  }

  waitForInitialization();

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
  waitForInitialization();

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
  waitForInitialization();

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
  waitForInitialization();

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

  // If this node has a ConstantShapeBuffer, add its primary pointer (LongType* shape_info) to the set
  // This is what ShapeCacheLifecycleTracker uses to track allocations
  ConstantShapeBuffer* buffer = node->buffer();
  if (buffer != nullptr && buffer->primary() != nullptr) {
    out_pointers.insert(buffer->primary());
  }

  // Recursively collect from all children
  for (const auto* child : node->children()) {
    collectCachedPointers(child, out_pointers);
  }
}

}  // namespace sd
