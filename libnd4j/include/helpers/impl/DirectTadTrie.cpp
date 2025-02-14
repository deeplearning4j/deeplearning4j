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

#include "array/TadCalculator.h"

namespace sd {

thread_local DirectTadTrie::ThreadCache DirectTadTrie::_threadCache;




void TadTrieNode::setPack(TadPack* pack) {
    TadPack* old = _tadPack;
    _tadPack = pack;
    if (old) delete old;
}

bool DirectTadTrie::dimensionsEqual(const std::vector<LongType>& a, const std::vector<LongType>& b) const {
    return a == b;  // Vectors should already be sorted
}


bool DirectTadTrie::exists(const std::vector<LongType>& dimensions) const {
    const size_t stripeIdx = computeStripeIndex(dimensions);
    TAD_LOCK_TYPE<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

    const TadTrieNode* current = _roots[stripeIdx].get();

    // First level: dimension length
    for (const auto& child : current->children()) {
      if (child->value() == static_cast<LongType>(dimensions.size()) && !child->isDimension()) {
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
        if (child->value() == dimensions[i] && child->isDimension()) {
          current = child.get();
          found = true;
          break;
        }
      }
      if (!found) return false;
    }

    return current->pack() != nullptr;  // Changed from atomic load
}

std::vector<LongType> DirectTadTrie::sortDimensions(const std::vector<LongType>& dimensions) const {
    std::vector<LongType> sorted = dimensions;
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

TadPack* DirectTadTrie::search(const std::vector<LongType>& dimensions, size_t stripeIdx) const {
    std::shared_lock<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

    const TadTrieNode* current = _roots[stripeIdx].get();

    // First level: dimension length
    current = findChild(current, dimensions.size(), 0, false);
    if (!current) return nullptr;

    // Second level: dimensions
    for (size_t i = 0; i < dimensions.size(); i++) {
        current = findChild(current, dimensions[i], i + 1, true);
        if (!current) return nullptr;
    }

    return current->pack();  // Changed from atomic lo
}


TadPack* DirectTadTrie::insert(std::vector<LongType>& dimensions, LongType* originalShape) {
    TadTrieNode* current = _roots[computeStripeIndex(dimensions)].get();

    // First level: dimension length
    current = current->findOrCreateChild(dimensions.size(), 0, false);  // level 0 for length node
    if (!current) return nullptr;

    // Second level: dimensions
    for (size_t i = 0; i < dimensions.size(); i++) {
        current = current->findOrCreateChild(dimensions[i], i + 1, true);  // level i+1 for dimension nodes
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


const TadTrieNode* DirectTadTrie::findChild(const TadTrieNode* node, LongType value, int level, bool isDimension) const {
    if (!node) return nullptr;

    for (const auto& child : node->children()) {
        if (child->value() == value &&
            child->level() == level &&
            child->isDimension() == isDimension) {
          return child.get();
        }
    }
    
    return nullptr;
}

}  // namespace sd