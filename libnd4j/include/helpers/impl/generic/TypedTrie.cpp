/* ******************************************************************************
 *
 * Copyright (c) 2024 Konduit K.K.
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <helpers/generic/TypedTrie.h>

#include "array/ConstantShapeBuffer.h"
#include "array/TadPack.h"
#include "helpers/DirectTadTrie.h"

namespace sd {
namespace generic {

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
struct TypedTrie<KeyType, ValueType, NUM_STRIPES>::TrieNode {
  std::shared_ptr<ValueType> value;
  std::unordered_map<typename KeyType::value_type, std::unique_ptr<TrieNode>> children;
  std::atomic<uint32_t> refCount{0};
  std::atomic<bool> isComplete{false};
  std::chrono::steady_clock::time_point lastAccess;

  TrieNode() : lastAccess(std::chrono::steady_clock::now()) {}

  void incrementRef() {
    refCount.fetch_add(1, std::memory_order_acq_rel);
  }

  bool decrementRef() {
    return refCount.fetch_sub(1, std::memory_order_acq_rel) == 1;
  }
};



template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
TypedTrie<KeyType, ValueType, NUM_STRIPES>::TypedTrie()
 {
  try {
    for (auto& root : _roots) {
      root = std::make_unique<TrieNode>();
      _resourceManager.registerNode();
    }
  } catch (...) {
    cleanup();
    throw;
  }
}

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
TypedTrie<KeyType, ValueType, NUM_STRIPES>::~TypedTrie() {

}

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
size_t TypedTrie<KeyType, ValueType, NUM_STRIPES>::getStripeIndex(const KeyType& key) const {
  size_t h = 0;
  for (const auto& elem : key) {
    h = h * 31 + std::hash<typename KeyType::value_type>{}(elem);
  }
  return h & (NUM_STRIPES - 1);
}

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
typename TypedTrie<KeyType, ValueType, NUM_STRIPES>::TrieNode*
TypedTrie<KeyType, ValueType, NUM_STRIPES>::findOrCreateNode(TrieNode* root,
                                                             const KeyType& key,
                                                             bool createIfMissing) const {
  if (!root) return nullptr;

  TrieNode* current = root;
  for (const auto& k : key) {
    auto it = current->children.find(k);
    if (it == current->children.end()) {
      if (!createIfMissing) return nullptr;
      auto newNode = std::make_unique<TrieNode>();
      if (!newNode) return nullptr;
      current->children[k] = std::move(newNode);
    }
    current = current->children[k].get();
    if (!current) return nullptr;
  }
  return current;
}

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
void TypedTrie<KeyType, ValueType, NUM_STRIPES>::cleanupNode(TrieNode* node) {
  if (!node) return;

  auto now = std::chrono::steady_clock::now();
  auto age = std::chrono::duration_cast<std::chrono::minutes>(
      now - node->lastAccess).count();

  if (age > 30 && node->refCount.load(std::memory_order_acquire) == 0) {
    node->value.reset();

    for (auto it = node->children.begin(); it != node->children.end();) {
      auto* child = it->second.get();
      if (child) {
        cleanupNode(child);
        if (!child->value && child->children.empty()) {
          _resourceManager.unregisterNode();
          it = node->children.erase(it);
          continue;
        }
      }
      ++it;
    }
  }
}

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
std::shared_ptr<ValueType> TypedTrie<KeyType, ValueType, NUM_STRIPES>::get(const KeyType& key) const {
  if (!key.empty()) {
    auto scope = _resourceManager.createScope();
    size_t stripe = getStripeIndex(key);

    _locks.lockStripe(stripe, false);
    auto node = findOrCreateNode(_roots[stripe].get(), key, false);
    if (node && node->isComplete.load(std::memory_order_acquire)) {
      auto result = node->value;
      if (result) {
        node->lastAccess = std::chrono::steady_clock::now();
        _locks.unlockStripe(stripe, false);
        return result;
      }
    }
    _locks.unlockStripe(stripe, false);
  }
  return nullptr;
}


template std::shared_ptr<sd::ConstantShapeBuffer*>
sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
    sd::ConstantShapeBuffer*,
    32>::get(const std::vector<long long, std::allocator<long long>>& key) const;



template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
bool TypedTrie<KeyType, ValueType, NUM_STRIPES>::insert(const KeyType& key,
                                                        std::shared_ptr<ValueType> value) {
  if (!value || key.empty()) return false;

  auto scope = _resourceManager.createScope();
  size_t stripe = getStripeIndex(key);

  _locks.lockStripe(stripe, true);
  auto node = findOrCreateNode(_roots[stripe].get(), key, true);
  if (!node) {
    _locks.unlockStripe(stripe, true);
    return false;
  }

  if (node->value || node->isComplete.load(std::memory_order_acquire)) {
    _locks.unlockStripe(stripe, true);
    return false;
  }

  node->incrementRef();
  node->value = value;
  node->lastAccess = std::chrono::steady_clock::now();
  node->isComplete.store(true, std::memory_order_release);
  _locks.unlockStripe(stripe, true);
  return true;
}

template bool
sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
    sd::ConstantShapeBuffer*,
    32>::insert(const std::vector<long long, std::allocator<long long>>& key,
                std::shared_ptr<sd::ConstantShapeBuffer*> value);

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
bool TypedTrie<KeyType, ValueType, NUM_STRIPES>::remove(const KeyType& key) {
  auto scope = _resourceManager.createScope();
  size_t stripe = getStripeIndex(key);

  _locks.lockStripe(stripe, true);

  auto node = findOrCreateNode(_roots[stripe].get(), key, false);
  if (!node || !node->value) {
    _locks.unlockStripe(stripe, true);
    return false;
  }

  node->value.reset();
  node->isComplete.store(false, std::memory_order_release);

  _locks.unlockStripe(stripe, true);
  return true;
}

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
void TypedTrie<KeyType, ValueType, NUM_STRIPES>::cleanup() {
  auto scope = _resourceManager.createScope();
  std::vector<size_t> stripes;
  for (size_t i = 0; i < NUM_STRIPES; ++i) {
    stripes.push_back(i);
  }
  auto guard = _locks.acquireMultiLock(stripes, true);

  for (auto& root : _roots) {
    if (root) cleanupNode(root.get());
  }
}

template void
sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
    sd::ConstantShapeBuffer*,
    32>::cleanup();
template void
sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
    sd::TadPack*,
    32>::cleanup();

template
sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
    sd::ConstantShapeBuffer*,
    32>::~TypedTrie();

template<typename KeyType, typename ValueType, size_t NUM_STRIPES>
typename TypedTrie<KeyType, ValueType, NUM_STRIPES>::Stats
TypedTrie<KeyType, ValueType, NUM_STRIPES>::getStats() const {
  Stats stats;
  for (size_t i = 0; i < NUM_STRIPES; ++i) {
    stats.stripeCounts[i] = _locks.getStripeCount(i);
  }
  return stats;
}

}  // namespace generic
}  // namespace sd

template class sd::generic::TypedTrie<std::vector<unsigned char>, std::shared_ptr<sd::ConstantShapeBuffer>, 32>;
template class sd::generic::TypedTrie<std::vector<sd::LongType>, std::shared_ptr<sd::TadPack>, 32>;
template std::shared_ptr<sd::TadPack*>
sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
                       sd::TadPack*,
                       32>::get(const std::vector<long long, std::allocator<long long>>& key) const;
// Add to TypedTrie.cpp
template bool
sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
                       sd::TadPack*,
                       32>::insert(const std::vector<long long, std::allocator<long long>>& key,
                                   std::shared_ptr<sd::TadPack*> value);
template
    sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
                           sd::TadPack*,
                           32>::~TypedTrie();



template sd::generic::TypedTrie<std::vector<long long, std::allocator<long long>>,
                   sd::ConstantShapeBuffer*,
                   32ul>::TypedTrie();