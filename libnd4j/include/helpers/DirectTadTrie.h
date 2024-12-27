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

#ifndef LIBND4J_DIRECTTADTRIE_H
#define LIBND4J_DIRECTTADTRIE_H

#include <array/TadPack.h>
#include <system/common.h>
#include <vector>
#include <memory>
#include <shared_mutex>
#include <atomic>
#include <array>

namespace sd {
#ifndef __JAVACPP_HACK__

// Add make_unique implementation for pre-C++14
template<typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


class SD_LIB_EXPORT TadTrieNode {
private:
    std::vector<std::unique_ptr<TadTrieNode>> _children;
    LongType _value;
    int _level;
    bool _isDimension;
    std::atomic<int> _cachedEntries;  // Count of cached entries in this subtree

public:
    std::atomic<TadPack*> _tadPack;  // Made public for direct access

    TadTrieNode(LongType value = 0, int level = 0, bool isDimension = true)
        : _value(value), _level(level), _isDimension(isDimension), _cachedEntries(0), _tadPack(nullptr) {}

    ~TadTrieNode() = default;

    TadTrieNode* findOrCreateChild(LongType value, int level, bool isDimension) {
        for (auto& child : _children) {
            if (child->value() == value && child->isDimension() == isDimension) {
                return child.get();
            }
        }

        auto newNode = make_unique_helper<TadTrieNode>(value, level, isDimension);
        auto* ptr = newNode.get();
        _children.push_back(std::move(newNode));
        return ptr;
    }

    const std::vector<std::unique_ptr<TadTrieNode>>& children() const { return _children; }
    LongType value() const { return _value; }
    int level() const { return _level; }
    bool isDimension() const { return _isDimension; }

    void incrementCachedEntries() { _cachedEntries.fetch_add(1, std::memory_order_relaxed); }
    void decrementCachedEntries() { _cachedEntries.fetch_sub(1, std::memory_order_relaxed); }
    int getCachedEntries() const { return _cachedEntries.load(std::memory_order_relaxed); }
    void setPack(TadPack* pack);
    TadPack* pack() const;

};



#if __cplusplus >= 201703L
#define TAD_MUTEX_TYPE std::shared_mutex
#define TAD_LOCK_TYPE std::shared_lock
#else
#define TAD_MUTEX_TYPE std::mutex
#define TAD_LOCK_TYPE std::lock_guard
#endif

class SD_LIB_EXPORT DirectTadTrie {
private:
    static const size_t NUM_STRIPES = 32;
    std::array<std::unique_ptr<TadTrieNode>, NUM_STRIPES> _roots;
    mutable std::array<TAD_MUTEX_TYPE, NUM_STRIPES> _mutexes = {};
    std::array<std::atomic<int>, NUM_STRIPES> _stripeCounts = {};

    struct ThreadCache {
        static const size_t CACHE_SIZE = 1024;
        std::vector<std::pair<std::vector<LongType>, TadPack*>> entries;
        ThreadCache() { entries.reserve(CACHE_SIZE); }
    };
    int countEntriesInSubtree(const TadTrieNode* node) const;

    static thread_local ThreadCache _threadCache;

public:
    // Proper constructors for move/copy
    DirectTadTrie() {
        for (size_t i = 0; i < NUM_STRIPES; i++) {
            _roots[i] =  make_unique_helper<TadTrieNode>();
            _stripeCounts[i].store(0, std::memory_order_relaxed);
        }
    }

    DirectTadTrie(const DirectTadTrie&) = delete;  // Prevent copying
    DirectTadTrie& operator=(const DirectTadTrie&) = delete;

    DirectTadTrie(DirectTadTrie&&) = default;  // Allow moving
    DirectTadTrie& operator=(DirectTadTrie&&) = default;

    size_t computeStripeIndex(const std::vector<LongType>& dimensions) const {
        size_t hash = 0;
        for (auto dim : dimensions) {
            hash = hash * 31 + static_cast<size_t>(dim);
        }
        return hash % NUM_STRIPES;
    }

    TadTrieNode* getOrCreateNode(const std::vector<LongType>& dimensions) {
        const size_t stripeIdx = computeStripeIndex(dimensions);
        std::unique_lock<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

        TadTrieNode* current = _roots[stripeIdx].get();

        // First level: dimension length
        current = current->findOrCreateChild(dimensions.size(), 0, false);

        // Second level: dimensions
        for (size_t i = 0; i < dimensions.size(); i++) {
            current = current->findOrCreateChild(dimensions[i], i + 1, true);
        }

        return current;
    }

    TadPack* getOrCreate(const std::vector<LongType>& dimensions) {
        // Check thread-local cache first
        for (const auto& entry : _threadCache.entries) {
            if (entry.first == dimensions) {
                return entry.second;
            }
        }

        auto* node = getOrCreateNode(dimensions);
        return node->_tadPack.load(std::memory_order_acquire);
    }

    void incrementStripeCount(size_t stripeIdx) {
        _stripeCounts[stripeIdx].fetch_add(1, std::memory_order_relaxed);
    }

    void decrementStripeCount(size_t stripeIdx) {
        _stripeCounts[stripeIdx].fetch_sub(1, std::memory_order_relaxed);
    }

    int totalCachedEntries() const {
        int total = 0;
        for (size_t i = 0; i < NUM_STRIPES; i++) {
            total += _stripeCounts[i].load(std::memory_order_relaxed);
        }
        return total;
    }

    bool exists(const std::vector<LongType>& dimensions) const {
        const size_t stripeIdx = computeStripeIndex(dimensions);
        TAD_LOCK_TYPE<TAD_MUTEX_TYPE> lock(_mutexes[stripeIdx]);

        const TadTrieNode* current = _roots[stripeIdx].get();

        // First level: dimension length
        for (const auto& child : current->children()) {
            if (child->value() == dimensions.size() && !child->isDimension()) {
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

        return current->_tadPack.load(std::memory_order_acquire) != nullptr;
    }
    TadPack* search(const std::vector<LongType>& dimensions, size_t stripeIdx) const;
    std::vector<LongType> sortDimensions(const std::vector<LongType>& dimensions) const;
    void updateThreadCache(const std::vector<LongType>& dimensions, TadPack* pack);
    bool dimensionsEqual(const std::vector<LongType>& a, const std::vector<LongType>& b) const;
    size_t getStripeIndex(const std::vector<LongType>& dimensions) const;
    size_t computeHash(const std::vector<LongType>& dimensions) const;
    TadPack* insert(const std::vector<LongType>& dimensions, size_t stripeIdx);
    const TadTrieNode* findChild(const TadTrieNode* node, LongType value, int level, bool isDimension) const;
};

}  // namespace sd
#endif
#endif //LIBND4J_DIRECTTADTRIE_H