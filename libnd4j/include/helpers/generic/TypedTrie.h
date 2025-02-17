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

#ifndef SD_GENERIC_TYPED_TRIE_H_
#define SD_GENERIC_TYPED_TRIE_H_

#include <system/common.h>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "StripedLocks.h"
#include "ResourceManager.h"
namespace sd {
namespace generic {

template<typename KeyType, typename ValueType, size_t NUM_STRIPES = 32>
class SD_LIB_EXPORT TypedTrie {
 private:
  struct TrieNode;


  std::array<std::unique_ptr<TrieNode>, NUM_STRIPES> _roots;
  ResourceManager _resourceManager;

  size_t getStripeIndex(const KeyType& key) const;
  TrieNode* findOrCreateNode(TrieNode* root, const KeyType& key, bool createIfMissing) const;
  void cleanupNode(TrieNode* node);

 public:
  ~TypedTrie();
  TypedTrie();

  // Delete copy constructor and assignment
  TypedTrie& operator=(const TypedTrie&) = delete;

  // Allow move operations
  TypedTrie(TypedTrie&&) = default;
  TypedTrie& operator=(TypedTrie&&) = default;


  std::shared_ptr<ValueType> get(const KeyType& key) const;
  bool insert(const KeyType& key, std::shared_ptr<ValueType> value);
  bool remove(const KeyType& key);
  void cleanup();

  struct Stats {
    size_t totalNodes;
    size_t activeOperations;
    std::array<uint32_t, NUM_STRIPES> stripeCounts;
  };

  Stats getStats() const;
  StripedLocks<NUM_STRIPES> _locks;
};

}  // namespace generic
}  // namespace sd

#endif  // SD_GENERIC_TYPED_TRIE_H_