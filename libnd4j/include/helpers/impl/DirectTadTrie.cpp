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

#include "../DirectTadTrie.h"

#include <array/TadPack.h>

#include <algorithm>
#include <memory>
#include <atomic>
#include <sstream>
#include <string>
#include <unordered_set>

#include "array/TadCalculator.h"

namespace sd {

std::shared_ptr<TadPack> DirectTadTrie::enhancedSearch(const std::vector<LongType>& dimensions, LongType* originalShape, size_t stripeIdx) {
  const TadTrieNode* current = _roots[stripeIdx].get();
  int rank = shape::rank(originalShape);

  // Navigate to dimension length node
  current = findChild(current, dimensions.size(), 0, false, rank);
  if (!current) return nullptr;

  // Navigate through dimension nodes
  for (size_t i = 0; i < dimensions.size(); i++) {
    current = findChild(current, dimensions[i], i + 1, true, rank);
    if (!current) return nullptr;
  }

  // Found a matching node, now verify TadPack compatibility
  std::shared_ptr<TadPack> pack = current->pack();
  if (!pack) return nullptr;

  // Use cached signature for fast comparison - no TadCalculator needed!
  const TadPackSignature* signature = current->packSignature();
  if (!signature) {
    // Signature not cached (shouldn't happen, but handle gracefully)
    return nullptr;
  }

  // Fast comparison using cached signature instead of creating TadCalculator
  if (!signature->matches(originalShape)) {
    return nullptr;
  }

  return pack;
}

// Enhanced stride-aware hash computation
size_t DirectTadTrie::computeStrideAwareHash(const std::vector<LongType>& dimensions, LongType* originalShape)  {
  if (!originalShape) return 0;

  size_t hash = 17; // Prime number starting point

  // Handle empty dimensions specially
  if (dimensions.empty()) {
    // Empty dimensions case - hash based on shape only
    hash = hash * 31 + 0; // Marker for empty dimensions
  } else {
    // Add dimension-specific hash contribution with position-dependence
    for (size_t i = 0; i < dimensions.size(); i++) {
      hash = hash * 31 + static_cast<size_t>(dimensions[i]) * (i + 1);
    }
  }

  // Add rank - critical for distinguishing different dimension arrays
  int rank = shape::rank(originalShape);
  hash = hash * 13 + rank * 19;

  // Add shape signature based on shape dimensions with position-dependence
  LongType* shapeInfo = shape::shapeOf(originalShape);
  for (int i = 0; i < rank; i++) {
    hash = hash * 17 + static_cast<size_t>(shapeInfo[i]) * (11 + i);
  }

  // Add stride information to make the hash more specific
  LongType* strides = shape::stride(originalShape);
  for (int i = 0; i < rank; i++) {
    hash = hash * 23 + static_cast<size_t>(strides[i]) * (7 + i);
  }

  // Add total element count to distinguish differently sized arrays
  hash = hash * 41 + shape::length(originalShape);

  // Add data type and order information
  hash = hash * 29 + static_cast<size_t>(ArrayOptions::dataType(originalShape));
  hash = hash * 37 + static_cast<size_t>(shape::order(originalShape));

  // Compute the final stripe index
  return hash % NUM_STRIPES;
}

bool DirectTadTrie::exists(const std::vector<LongType>& dimensions, LongType* originalShape)  {
  if (!originalShape) return false;

  const size_t stripeIdx = computeStripeIndex(dimensions, originalShape);
  SHARED_LOCK_TYPE<MUTEX_TYPE> lock(_mutexes[stripeIdx]);

  // Using the enhanced search method which verifies TadPack compatibility
  return enhancedSearch(dimensions, originalShape, stripeIdx) != nullptr;
}

std::vector<LongType> DirectTadTrie::sortDimensions(const std::vector<LongType>& dimensions) const {
  std::vector<LongType> sorted = dimensions;
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

std::shared_ptr<TadPack> DirectTadTrie::search(const std::vector<LongType>& dimensions, int originalShapeRank, size_t stripeIdx) const {
  // No need for locking - caller handles locking (e.g., in getOrCreate)
  const TadTrieNode* current = _roots[stripeIdx].get();

  // First level: dimension length
  current = findChild(current, dimensions.size(), 0, false, originalShapeRank);
  if (!current) return nullptr;

  // Second level: dimensions
  for (size_t i = 0; i < dimensions.size(); i++) {
    current = findChild(current, dimensions[i], i + 1, true, originalShapeRank);
    if (!current) return nullptr;
  }

  return current->pack();
}

std::shared_ptr<TadPack> DirectTadTrie::getOrCreate(std::vector<LongType>& dimensions, LongType* originalShape) {
  if (!originalShape) {
    THROW_EXCEPTION("Original shape cannot be null in TAD calculation");
  }

  // Use the enhanced hash computation for better distribution
  const size_t stripeIdx = computeStrideAwareHash(dimensions, originalShape);

  // First try a read-only lookup
  {
    SHARED_LOCK_TYPE<MUTEX_TYPE> readLock(_mutexes[stripeIdx]);
    std::shared_ptr<TadPack> existing = enhancedSearch(dimensions, originalShape, stripeIdx);
    if (existing) {
      return existing;
    }
  }

  // If not found, use insert which will handle the write lock
  return insert(dimensions, originalShape);
}

std::shared_ptr<TadPack> DirectTadTrie::insert(std::vector<LongType>& dimensions, LongType* originalShape) {
  if (!originalShape) {
    THROW_EXCEPTION("Original shape cannot be null in TAD calculation");
  }

  int rank = shape::rank(originalShape);
  // Use the enhanced hash computation for better distribution
  const size_t stripeIdx = computeStrideAwareHash(dimensions, originalShape);
  // Use exclusive lock for write operation (inserting new TAD packs)
  EXCLUSIVE_LOCK_TYPE<MUTEX_TYPE> lock(_mutexes[stripeIdx]);

  // Check if a compatible TadPack already exists
  std::shared_ptr<TadPack> existing = enhancedSearch(dimensions, originalShape, stripeIdx);
  if (existing) {
    return existing;
  }

  // No compatible TadPack found, create a new one
  TadTrieNode* current = _roots[stripeIdx].get();

  // First level: dimension length node with shape rank
  current = current->findOrCreateChild(dimensions.size(), 0, false, rank);
  if (!current) {
    THROW_EXCEPTION("Failed to create dimension length node");
  }

  // Second level: dimension nodes with shape rank
  for (size_t i = 0; i < dimensions.size(); i++) {
    current = current->findOrCreateChild(dimensions[i], i + 1, true, rank);
    if (!current) {
      THROW_EXCEPTION("Failed to create dimension node");
    }
  }

  // Create the TadPack only if it doesn't exist yet
  if (!current->pack()) {
    TadCalculator *calculator = nullptr;
    std::shared_ptr<TadPack> newPack;

    try {
      calculator = new TadCalculator(originalShape);
      calculator->createTadPack(dimensions);

      // Create a new TadPack with full dimension information
      // Use releaseOffsets() to transfer ownership of the offsets buffer to TadPack
      // Wrap in shared_ptr for proper memory management
      newPack = std::make_shared<TadPack>(
          calculator->tadShape(),
          calculator->releaseOffsets(),  // Transfer ownership
          calculator->numberOfTads(),
          dimensions.data(),
          dimensions.size());

      // Store the TadPack in the node
      // setPack now also caches the signature for future fast comparisons
      current->setPack(newPack);

      // Clean up the calculator (safe now that offsets ownership was transferred)
      delete calculator;
      calculator = nullptr;

    } catch (const std::exception& e) {
      // Clean up on exception to prevent memory leaks
      // shared_ptr will automatically clean up newPack
      if (calculator != nullptr) {
        delete calculator;
        calculator = nullptr;
      }
      std::string msg = "TAD creation failed: ";
      msg += e.what();
      THROW_EXCEPTION(msg.c_str());
    }
  }

  return current->pack();
}


const TadTrieNode* DirectTadTrie::findChild(const TadTrieNode* node, LongType value, int level, bool isDimension, int shapeRank) const {
  if (!node) return nullptr;

  for (const auto& child : node->children()) {
    if (child->value() == value &&
        child->level() == level &&
        child->isDimension() == isDimension &&
        child->shapeRank() == shapeRank) {
      return child.get();
    }
  }

  return nullptr;
}

// Helper function to recursively delete TadPacks from a node and its children
// This ensures TadPack destructors are called, which triggers recordDeallocation()
static void deleteTadPacksRecursive(TadTrieNode* node, int& deletedCount) {
  if (!node) return;

  // First, recursively delete from all children
  const auto& children = node->children();
  for (const auto& child : children) {
    deleteTadPacksRecursive(child.get(), deletedCount);
  }

  // Then delete this node's TadPack if it exists
  // shared_ptr will handle deletion automatically when we reset it
  auto pack = node->pack();
  if (pack) {
    // Count and log deletion for verification
    deletedCount++;
    // DEBUG: Log TAD pack deletion with detailed info
    sd_printf("DirectTadTrie: Deleting TadPack %p (count: %d)\n", pack.get(), deletedCount);

    // Clear the shared_ptr to trigger TadPack destructor
    // The destructor will call TADCacheLifecycleTracker::recordDeallocation()
    // if SD_GCC_FUNCTRACE is defined during compilation
    node->setPack(nullptr);
  }
}

void DirectTadTrie::clear() {
  // Clear all stripes
  // NOTE: Removed #ifndef __JAVACPP_HACK__ guard to fix TAD cache memory leak
  // The guard was preventing cache cleanup when JavaCPP is used (production mode)
  // This caused indefinite accumulation of TADPack objects despite clearTADCache() calls

  int totalDeleted = 0;
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    // Use exclusive lock for write operation (clearing the cache)
    EXCLUSIVE_LOCK_TYPE<MUTEX_TYPE> lock(_mutexes[i]);

    // CRITICAL FIX: Explicitly delete all TadPacks before recreating roots
    // This ensures TadPack destructors are called, which invokes recordDeallocation()
    // for proper lifecycle tracking.
    //
    // IMPORTANT: We CANNOT rely on unique_ptr cascade deletion because:
    // 1. TadTrieNode destructor deletes _tadPack only if SD_GCC_FUNCTRACE is defined
    // 2. Functrace may be auto-disabled during build, causing guards to evaluate false
    // 3. Even if guards pass, destructor might not run if roots are replaced before going out of scope
    //
    // By explicitly calling deleteTadPacksRecursive() BEFORE replacing roots,
    // we guarantee that:
    // - All TadPack objects are explicitly deleted via delete operator
    // - Their destructors run and call recordDeallocation() (if tracking enabled)
    // - Pointers are cleared to nullptr to prevent double-delete in node destructors
    int deletedCount = 0;
    deleteTadPacksRecursive(_roots[i].get(), deletedCount);
    totalDeleted += deletedCount;

    // Recreate the root node - this will delete the old tree structure
    // (nodes are already cleaned of TadPacks above via deleteTadPacksRecursive)
    // The old root's unique_ptr goes out of scope here, triggering node destructor cascade
    // But TadPacks are already deleted and nulled out, so no double-delete occurs
    _roots[i] = std::make_unique<TadTrieNode>(0, 0, false);
    _stripeCounts[i].store(0);
  }

  // Reset current counters (but preserve peak values for diagnostics)
  _current_entries.store(0);
  _current_bytes.store(0);
}

void DirectTadTrie::countEntriesAndBytes(const TadTrieNode* node, LongType& entries, LongType& bytes) const {
  if (node == nullptr) return;

  // If this node has a TadPack, count it
  auto pack = node->pack();
  if (pack != nullptr) {
    entries++;

    // Calculate total bytes for this TadPack
    // Shape info buffer
    const LongType* shapeInfo = pack->primaryShapeInfo();
    if (shapeInfo != nullptr) {
      LongType shapeInfoLength = shape::shapeInfoLength(shapeInfo);
      bytes += shapeInfoLength * sizeof(LongType);
    }

    // Offsets buffer
    const LongType* offsets = pack->primaryOffsets();
    if (offsets != nullptr) {
      LongType numTads = pack->numberOfTads();
      bytes += numTads * sizeof(LongType);
    }
  }

  // Recursively count children
  const std::vector<std::unique_ptr<TadTrieNode>>& children = node->children();
  for (const auto& child : children) {
    countEntriesAndBytes(child.get(), entries, bytes);
  }
}

LongType DirectTadTrie::getCachedEntries() const {
  LongType total_entries = 0;
  LongType total_bytes = 0;

  // Count entries across all stripes
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    // Lock this stripe for reading
    SHARED_LOCK_TYPE<MUTEX_TYPE> lock(_mutexes[i]);

    const TadTrieNode* root = _roots[i].get();
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

LongType DirectTadTrie::getCachedBytes() const {
  // getCachedEntries() updates both entries and bytes
  getCachedEntries();
  return _current_bytes.load();
}

LongType DirectTadTrie::getPeakCachedEntries() const {
  return _peak_entries.load();
}

LongType DirectTadTrie::getPeakCachedBytes() const {
  return _peak_bytes.load();
}

void DirectTadTrie::buildStringRepresentation(const TadTrieNode* node, std::stringstream& ss,
                                              const std::string& indent, int currentDepth,
                                              int maxDepth, int& entriesShown, int maxEntries) const {
  if (node == nullptr) return;
  if (maxDepth != -1 && currentDepth > maxDepth) return;
  if (maxEntries != -1 && entriesShown >= maxEntries) return;

  // Check if this node has a TadPack
  auto pack = node->pack();
  if (pack != nullptr) {
    entriesShown++;

    // Display node info
    ss << indent << "Node[level=" << node->level()
       << ", value=" << node->value()
       << ", isDim=" << (node->isDimension() ? "true" : "false")
       << ", rank=" << node->shapeRank()
       << "]\n";

    // Display TAD pack details
    const LongType* shapeInfo = pack->primaryShapeInfo();
    if (shapeInfo != nullptr) {
      int rank = shape::rank(shapeInfo);
      ss << indent << "  TAD Shape: rank=" << rank
         << ", order=" << shape::order(shapeInfo)
         << ", dtype=" << DataTypeUtils::asString(ArrayOptions::dataType(shapeInfo)) << "\n";

      // Display TAD dimensions
      ss << indent << "  TAD Dims: [";
      const LongType* dims = shape::shapeOf(shapeInfo);
      for (int i = 0; i < rank; i++) {
        if (i > 0) ss << ", ";
        ss << dims[i];
      }
      ss << "]\n";

      // Display TAD strides
      ss << indent << "  TAD Strides: [";
      const LongType* strides = shape::stride(shapeInfo);
      for (int i = 0; i < rank; i++) {
        if (i > 0) ss << ", ";
        ss << strides[i];
      }
      ss << "]\n";
    }

    // Display number of TADs and offset info
    LongType numTads = pack->numberOfTads();
    ss << indent << "  Number of TADs: " << numTads << "\n";

    // Display memory usage
    LongType shapeInfoBytes = 0;
    LongType offsetsBytes = 0;
    if (shapeInfo != nullptr) {
      LongType shapeInfoLength = shape::shapeInfoLength(shapeInfo);
      shapeInfoBytes = shapeInfoLength * sizeof(LongType);
    }
    if (pack->primaryOffsets() != nullptr) {
      offsetsBytes = numTads * sizeof(LongType);
    }
    ss << indent << "  Memory: shape_info=" << shapeInfoBytes
       << " bytes, offsets=" << offsetsBytes
       << " bytes, total=" << (shapeInfoBytes + offsetsBytes) << " bytes\n";

    if (maxEntries != -1 && entriesShown >= maxEntries) {
      ss << indent << "  ... (max entries reached)\n";
      return;
    }
  }

  // Recursively process children
  const std::vector<std::unique_ptr<TadTrieNode>>& children = node->children();
  if (!children.empty() && (maxDepth == -1 || currentDepth < maxDepth)) {
    for (const auto& child : children) {
      if (maxEntries != -1 && entriesShown >= maxEntries) break;
      buildStringRepresentation(child.get(), ss, indent + "  ", currentDepth + 1,
                               maxDepth, entriesShown, maxEntries);
    }
  }
}

std::string DirectTadTrie::toString(int maxDepth, int maxEntries) const {
  std::stringstream ss;

  // Get current statistics
  LongType totalEntries = getCachedEntries();
  LongType totalBytes = getCachedBytes();
  LongType peakEntries = getPeakCachedEntries();
  LongType peakBytes = getPeakCachedBytes();

  // Header
  ss << "DirectTadTrie [" << NUM_STRIPES << " stripes]\n";
  ss << "Current: " << totalEntries << " entries, " << totalBytes << " bytes\n";
  ss << "Peak: " << peakEntries << " entries, " << peakBytes << " bytes\n";
  ss << "Showing: max depth=" << (maxDepth == -1 ? "unlimited" : std::to_string(maxDepth))
     << ", max entries=" << (maxEntries == -1 ? "unlimited" : std::to_string(maxEntries)) << "\n";
  ss << "---\n";

  int entriesShown = 0;

  // Traverse each stripe
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    // Lock this stripe for reading
    SHARED_LOCK_TYPE<MUTEX_TYPE> lock(_mutexes[i]);

    const TadTrieNode* root = _roots[i].get();
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

void DirectTadTrie::getCachedPointers(std::unordered_set<void*>& out_pointers) const {
  // Traverse all stripes and collect TadPack pointers
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    SHARED_LOCK_TYPE<MUTEX_TYPE> lock(_mutexes[i]);

    const TadTrieNode* root = _roots[i].get();
    if (root != nullptr) {
      collectCachedPointers(root, out_pointers);
    }
  }
}

void DirectTadTrie::collectCachedPointers(const TadTrieNode* node, std::unordered_set<void*>& out_pointers) const {
  if (node == nullptr) return;

  // If this node has a TadPack, add it to the set
  auto pack = node->pack();
  if (pack != nullptr) {
    out_pointers.insert(pack.get());
  }

  // Recursively collect from all children
  for (const auto& child : node->children()) {
    collectCachedPointers(child.get(), out_pointers);
  }
}

}  // namespace sd
