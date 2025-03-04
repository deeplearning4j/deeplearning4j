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

#include "array/TadCalculator.h"

namespace sd {

TadPack* DirectTadTrie::enhancedSearch(const std::vector<LongType>& dimensions, LongType* originalShape, size_t stripeIdx) {
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

  // Found a matching node, now verify TadPack compatibility with shape signature
  TadPack* pack = current->pack();
  if (!pack) return nullptr;

  // Get the stored shape info from the TadPack
  LongType* storedShapeInfo = pack->primaryShapeInfo();
  LongType* storedStrides = shape::stride(storedShapeInfo);
  int storedRank = shape::rank(storedShapeInfo);

  // Compare with expected strides and shape
  // Create a temporary calculator to check what the strides should be
  TadCalculator tempCalc(originalShape);
  tempCalc.createTadPack(dimensions);
  LongType* expectedShapeInfo = tempCalc.tadShape().primary();
  LongType* expectedStrides = shape::stride(expectedShapeInfo);

  // Check if strides are compatible
  for (int i = 0; i < storedRank; i++) {
    if (storedStrides[i] != expectedStrides[i]) {
      return nullptr; // Strides don't match, not the right TadPack
    }
  }

  // Additional verification for shape dimensions
  LongType* storedShape = shape::shapeOf(storedShapeInfo);
  LongType* expectedShape = shape::shapeOf(expectedShapeInfo);
  for (int i = 0; i < storedRank; i++) {
    if (storedShape[i] != expectedShape[i]) {
      return nullptr; // Shape dimensions don't match
    }
  }

  // Verify order and data type match
  if (shape::order(storedShapeInfo) != shape::order(expectedShapeInfo) ||
      ArrayOptions::dataType(storedShapeInfo) != ArrayOptions::dataType(expectedShapeInfo)) {
    return nullptr;
  }

  // If everything matches, return the found TadPack
  return pack;
}

// Enhanced stride-aware hash computation
size_t DirectTadTrie::computeStrideAwareHash(const std::vector<LongType>& dimensions, LongType* originalShape)  {
  if (!originalShape) return 0;

  size_t hash = 17; // Prime number starting point

  // Add dimension-specific hash contribution with position-dependence
  for (size_t i = 0; i < dimensions.size(); i++) {
    hash = hash * 31 + static_cast<size_t>(dimensions[i]) * (i + 1);
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

// Check if dimensions are compatible with a TadPack


bool DirectTadTrie::exists(const std::vector<LongType>& dimensions, LongType* originalShape)  {
  if (!originalShape) return false;

  const size_t stripeIdx = computeStripeIndex(dimensions, originalShape);
  TAD_LOCK_TYPE<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

  // Using the enhanced search method which verifies TadPack compatibility
  return enhancedSearch(dimensions, originalShape, stripeIdx) != nullptr;
}

std::vector<LongType> DirectTadTrie::sortDimensions(const std::vector<LongType>& dimensions) const {
  std::vector<LongType> sorted = dimensions;
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

TadPack* DirectTadTrie::search(const std::vector<LongType>& dimensions, int originalShapeRank, size_t stripeIdx) const {
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

// Critical method that needs revision for better thread safety
TadPack* DirectTadTrie::getOrCreate(std::vector<LongType>& dimensions, LongType* originalShape) {
  if (!originalShape) {
    THROW_EXCEPTION("Original shape cannot be null in TAD calculation");
  }

  // Use the enhanced hash computation for better distribution
  const size_t stripeIdx = computeStrideAwareHash(dimensions, originalShape);

  // First try a read-only lookup
  {
    TAD_LOCK_TYPE<TAD_MUTEX_TYPE> readLock(_mutexes[stripeIdx]);
    TadPack* existing = enhancedSearch(dimensions, originalShape, stripeIdx);
    if (existing) {
      return existing;
    }
  }

  // If not found, use insert which will handle the write lock
  return insert(dimensions, originalShape);
}

TadPack* DirectTadTrie::insert(std::vector<LongType>& dimensions, LongType* originalShape) {
  if (!originalShape) {
    THROW_EXCEPTION("Original shape cannot be null in TAD calculation");
  }

  int rank = shape::rank(originalShape);
  // Use the enhanced hash computation for better distribution
  const size_t stripeIdx = computeStrideAwareHash(dimensions, originalShape);
  std::unique_lock<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

  // Check if a compatible TadPack already exists
  TadPack* existing = enhancedSearch(dimensions, originalShape, stripeIdx);
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
    try {
      TadCalculator calculator(originalShape);
      calculator.createTadPack(dimensions);

      // Create a new TadPack with full dimension information
      TadPack* newPack = new TadPack(
          calculator.tadShape(),
          calculator.tadOffsets(),
          calculator.numberOfTads(),
          dimensions.data(),
          dimensions.size());

      // Store the TadPack in the node (setPack handles synchronization)
      current->setPack(newPack);

    } catch (const std::exception& e) {
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

}  // namespace sd