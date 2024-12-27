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
#include <algorithm>
#include <array/TadPack.h>

namespace sd {

thread_local DirectTadTrie::ThreadCache DirectTadTrie::_threadCache;



TadPack* TadTrieNode::pack() const {
    return _tadPack.load(std::memory_order_acquire);
}

void TadTrieNode::setPack(TadPack* pack) {
    if (pack != nullptr && _tadPack.load(std::memory_order_relaxed) == nullptr) {
        incrementCachedEntries();
    } else if (pack == nullptr && _tadPack.load(std::memory_order_relaxed) != nullptr) {
        decrementCachedEntries();
    }
    _tadPack.store(pack, std::memory_order_release);
}

size_t DirectTadTrie::computeHash(const std::vector<LongType>& dimensions) const {
    size_t hash = 0;
    for (auto dim : dimensions) {
        hash = hash * 31 + static_cast<size_t>(dim);
    }
    return hash;
}

size_t DirectTadTrie::getStripeIndex(const std::vector<LongType>& dimensions) const {
    return computeHash(dimensions) % NUM_STRIPES;
}

bool DirectTadTrie::dimensionsEqual(const std::vector<LongType>& a, const std::vector<LongType>& b) const {
    return a == b;  // Vectors should already be sorted
}

void DirectTadTrie::updateThreadCache(const std::vector<LongType>& dimensions, TadPack* pack) {
    if (_threadCache.entries.size() >= ThreadCache::CACHE_SIZE) {
        _threadCache.entries.clear();
    }
    _threadCache.entries.emplace_back(dimensions, pack);
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
    
    return current->pack();
}

TadPack* DirectTadTrie::insert(const std::vector<LongType>& dimensions, size_t stripeIdx) {
    std::unique_lock<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);
    
    // Double-check after acquiring exclusive lock
    if (TadPack* existing = search(dimensions, stripeIdx)) {
        return existing;
    }
    
    TadTrieNode* current = _roots[stripeIdx].get();
    
    // First level: dimension length
    current = current->findOrCreateChild(dimensions.size(), 0, false);
    
    // Second level: dimensions
    for (size_t i = 0; i < dimensions.size(); i++) {
        current = current->findOrCreateChild(dimensions[i], i + 1, true);
    }
    
    incrementStripeCount(stripeIdx);
    return current->pack();
}

const TadTrieNode* DirectTadTrie::findChild(const TadTrieNode* node, LongType value, 
                                          int level, bool isDimension) const {
    if (!node) return nullptr;
    
    for (const auto& child : node->children()) {
        if (child->value() == value && child->isDimension() == isDimension) {
            return child.get();
        }
    }
    
    return nullptr;
}

int DirectTadTrie::countEntriesInSubtree(const TadTrieNode* node) const {
    if (!node) return 0;
    int count = node->getCachedEntries();
    for (const auto& child : node->children()) {
        count += countEntriesInSubtree(child.get());
    }
    return count;
}

}  // namespace sd