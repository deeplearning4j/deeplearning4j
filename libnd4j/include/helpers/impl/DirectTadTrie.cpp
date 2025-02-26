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

void TadTrieNode::setPack(TadPack* pack) {
  if (!pack) return;

  // Use atomic compare-and-swap for thread safety
  TadPack* expectedNull = nullptr;
  if (_tadPack == nullptr &&
      __sync_bool_compare_and_swap(&_tadPack, expectedNull, pack)) {
    // Successfully set the pack
    return;
  } else if (_tadPack != nullptr) {
    // Pack is already set - DO NOTHING
    // Cleanup the unneeded new pack
    if (pack != _tadPack) {
      delete pack;  // Clean up the unneeded new pack
    }
  }
}

bool DirectTadTrie::dimensionsEqual(const std::vector<LongType>& a, const std::vector<LongType>& b) const {
    return a == b;  // Vectors should already be sorted
}

bool DirectTadTrie::exists(const std::vector<LongType>& dimensions, LongType* originalShape) const {
    const size_t stripeIdx = computeStripeIndex(dimensions, originalShape);
    int rank = shape::rank(originalShape);
    TAD_LOCK_TYPE<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

    const TadTrieNode* current = _roots[stripeIdx].get();

    // First level: dimension length
    for (const auto& child : current->children()) {
      if (child->value() == static_cast<LongType>(dimensions.size()) && 
          !child->isDimension() && 
          child->shapeRank() == rank) {
        current = child.get();
        goto found_length;
      }
    }
    return false;

found_length:
    // Second level: dimensions
    for (size_t i = 0; i < dimensions.size(); i++) {
      bool found = false;
      for (const auto& child : current->children()) {
        if (child->value() == dimensions[i] && 
            child->isDimension() && 
            child->shapeRank() == rank) {
          current = child.get();
          found = true;
          break;
        }
      }
      if (!found) return false;
    }

    return current->pack() != nullptr;
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

    int rank = shape::rank(originalShape);
    const size_t stripeIdx = computeStripeIndex(dimensions, originalShape);
    
    // First try a read-only lookup without obtaining a write lock
    {
        std::unique_lock<TAD_MUTEX_TYPE> readLock(_mutexes[stripeIdx]);
        TadPack* existing = search(dimensions, rank, stripeIdx);
        if (existing != nullptr) {
            // Verify dimensions match by using node path, which we already traversed in search
            return existing;
        }
    }
    
    // If not found, grab exclusive lock and try again
    std::unique_lock<TAD_MUTEX_TYPE> writeLock(_mutexes[stripeIdx]);
    
    // Check again under the write lock
    TadPack* existing = search(dimensions, rank, stripeIdx);
    if (existing != nullptr) {
        return existing;
    }
    
    // Not found, need to create a new TAD pack
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
    
    // Check if the node already has a TAD pack (another thread might have created it)
    if (TadPack* nodeExisting = current->pack()) {
        return nodeExisting;
    }
    
    // Create the TAD pack under the exclusive lock
    try {
        // Create calculator and TAD pack atomically
        TadCalculator calculator(originalShape);
        calculator.createTadPack(dimensions);
        
        TadPack* newPack = new TadPack(
            calculator.tadShape(),
            calculator.tadOffsets(),
            calculator.numberOfTads(),
            dimensions.data(),
            dimensions.size());
        
        // Set the pack in the node (setPack handles the synchronization)
        current->setPack(newPack);
        
        // Verify the node has a valid pack now 
        TadPack* result = current->pack();
        if (result == nullptr) {
            THROW_EXCEPTION("Failed to set TAD pack in node");
        }
        
        return result;
    } catch (const std::exception& e) {
        std::string msg = "TAD creation failed: ";
        msg += e.what();
        THROW_EXCEPTION(msg.c_str());
    }
}

TadPack* DirectTadTrie::insert(std::vector<LongType>& dimensions, LongType* originalShape) {
    // Note: This method is kept for compatibility but is no longer used directly.
    // getOrCreate should be used instead, which has proper synchronization.
    int rank = shape::rank(originalShape);
    const size_t stripeIdx = computeStripeIndex(dimensions, originalShape);
    std::unique_lock<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);
    
    TadTrieNode* current = _roots[stripeIdx].get();

    // First level: dimension length
    current = current->findOrCreateChild(dimensions.size(), 0, false, rank);
    if (!current) return nullptr;

    // Second level: dimensions
    for (size_t i = 0; i < dimensions.size(); i++) {
        current = current->findOrCreateChild(dimensions[i], i + 1, true, rank);
        if (!current) return nullptr;
    }

    // Create TAD pack if needed
    if (!current->pack()) {
        try {
          TadCalculator calculator(originalShape);
          calculator.createTadPack(dimensions);

          TadPack* newPack = new TadPack(
              calculator.tadShape(),
              calculator.tadOffsets(),
              calculator.numberOfTads(),
              dimensions.data(),
              dimensions.size());

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