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

#include <helpers/DirectShapeTrie.h>
#include <array/ConstantShapeBuffer.h>
#include <array/PrimaryPointerDeallocator.h>
#include <array/DataType.h>
#include <array/ArrayOptions.h>
#include <helpers/shape.h>
#include <system/common.h>

namespace sd {

thread_local DirectShapeTrie::ThreadCache DirectShapeTrie::_threadCache;

ShapeTrieNode::ShapeTrieNode(LongType value, int level, bool isShape)
    : _buffer(nullptr), _value(value), _level(level), _isShape(isShape) {
#if defined(SD_GCC_FUNCTRACE)
  this->st.load_here();
#endif
}

ShapeTrieNode::~ShapeTrieNode() {
  auto* buf = _buffer.load(std::memory_order_acquire);
  if (buf != nullptr) {
    delete buf;
  }
}

ShapeTrieNode* ShapeTrieNode::findOrCreateChild(LongType value, int level, bool isShape) {
  for (auto& child : _children) {
    if (child->_value == value && child->_level == level &&
        child->_isShape == isShape) {
      return child.get();
    }
  }

  auto newNode = std::make_unique<ShapeTrieNode>(value, level, isShape);
  ShapeTrieNode* nodePtr = newNode.get();
  _children.push_back(std::move(newNode));
  return nodePtr;
}

const std::vector<std::unique_ptr<ShapeTrieNode>>& ShapeTrieNode::children() const {
  return _children;
}

ConstantShapeBuffer* ShapeTrieNode::buffer() const {
  return _buffer.load(std::memory_order_acquire);
}

void ShapeTrieNode::setBuffer(ConstantShapeBuffer* buf) {
  auto* oldBuf = _buffer.exchange(buf, std::memory_order_acq_rel);
  if (oldBuf != nullptr) {
    delete oldBuf;
  }
}

LongType ShapeTrieNode::value() const { return _value; }
int ShapeTrieNode::level() const { return _level; }
bool ShapeTrieNode::isShape() const { return _isShape; }

void ShapeTrieNode::collectStoreStackTrace() {
#if defined(SD_GCC_FUNCTRACE)
  this->storeStackTrace = backward::StackTrace();
  this->storeStackTrace.load_here(32);
#endif
}

DirectShapeTrie::ThreadCache::ThreadCache() : entries() {
  entries.reserve(CACHE_SIZE);
}

DirectShapeTrie::DirectShapeTrie() {
  for (size_t i = 0; i < NUM_STRIPES; i++) {
    _roots[i] = std::make_unique<ShapeTrieNode>();
  }

}

size_t DirectShapeTrie::computeHash(const LongType* shapeInfo) const {
  size_t hash = 14695981039346656037ULL;
  const size_t len = shape::shapeInfoLength(shape::rank(shapeInfo));

  size_t i = 0;
  for (; i + 4 <= len; i += 4) {
    hash ^= static_cast<size_t>(shapeInfo[i]);
    hash *= 1099511628211ULL;
    hash ^= static_cast<size_t>(shapeInfo[i + 1]);
    hash *= 1099511628211ULL;
    hash ^= static_cast<size_t>(shapeInfo[i + 2]);
    hash *= 1099511628211ULL;
    hash ^= static_cast<size_t>(shapeInfo[i + 3]);
    hash *= 1099511628211ULL;
  }

  for (; i < len; i++) {
    hash ^= static_cast<size_t>(shapeInfo[i]);
    hash *= 1099511628211ULL;
  }

  return hash;
}

size_t DirectShapeTrie::getStripeIndex(const LongType* shapeInfo) const {
  return computeHash(shapeInfo) & (NUM_STRIPES - 1);
}

bool DirectShapeTrie::shapeInfoEqual(const LongType* a, const LongType* b) const {
  if (a == b) return true;
  if (a == nullptr || b == nullptr) return false;

  const int rankA = shape::rank(a);
  if (rankA != shape::rank(b)) return false;

  const int len = shape::shapeInfoLength(rankA);
  return std::memcmp(a, b, len * sizeof(LongType)) == 0;
}

void DirectShapeTrie::updateThreadCache(const LongType* shapeInfo, ConstantShapeBuffer* buffer) {
  auto& cache = _threadCache.entries;
  if (cache.size() >= ThreadCache::CACHE_SIZE) {
    cache.erase(cache.begin());
  }
  cache.emplace_back(shapeInfo, buffer);
}

ConstantShapeBuffer* DirectShapeTrie::createBuffer(const LongType* shapeInfo) {
  const int shapeInfoLength = shape::shapeInfoLength(shape::rank(shapeInfo));
  LongType* shapeCopy = new LongType[shapeInfoLength];
  std::memcpy(shapeCopy, shapeInfo, shapeInfoLength * sizeof(LongType));

  auto hPtr = std::make_shared<PointerWrapper>(shapeCopy,
                                               std::make_shared<PrimaryPointerDeallocator>());
  return new ConstantShapeBuffer(hPtr);
}

void DirectShapeTrie::validateShapeInfo(const LongType* shapeInfo) const {
  if (shapeInfo == nullptr) {
    THROW_EXCEPTION("Shape info cannot be null");
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
      THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
    }
  }

  if (ArrayOptions::dataType(shapeInfo) == UNKNOWN) {
    THROW_EXCEPTION("Shape info created with invalid data type");
  }

  char order = shape::order(shapeInfo);
  if (order != 'c' && order != 'f') {
    std::string errorMessage = "Invalid ordering in shape buffer: ";
    errorMessage += order;
    THROW_EXCEPTION(errorMessage.c_str());
  }
}

const ShapeTrieNode* DirectShapeTrie::findChild(const ShapeTrieNode* node, LongType value,
                                                int level, bool isShape) const {
  for (const auto& child : node->children()) {
    if (child->value() == value && child->level() == level &&
        child->isShape() == isShape) {
      return child.get();
    }
  }
  return nullptr;
}

ConstantShapeBuffer* DirectShapeTrie::search(const LongType* shapeInfo, size_t stripeIdx) const {
  const ShapeTrieNode* current = _roots[stripeIdx].get();
  if (!current) return nullptr;

  const int rank = shape::rank(shapeInfo);

  // Check rank
  current = findChild(current, rank, 0, true);
  if (!current) return nullptr;

  // Check datatype
  current = findChild(current, ArrayOptions::dataType(shapeInfo), 1, true);
  if (!current) return nullptr;

  // Check order
  current = findChild(current, shape::order(shapeInfo), 2, true);
  if (!current) return nullptr;

  // Check shape values
  const LongType* shape = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank && current; i++) {
    current = findChild(current, shape[i], 3 + i, true);
    if (!current) return nullptr;
  }

  // Check stride values
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank && current; i++) {
    current = findChild(current, strides[i], 3 + rank + i, false);
    if (!current) return nullptr;
  }

  return current ? current->buffer() : nullptr;
}

ConstantShapeBuffer* DirectShapeTrie::insert(const LongType* shapeInfo, size_t stripeIdx) {
  ShapeTrieNode* current = _roots[stripeIdx].get();
  const int rank = shape::rank(shapeInfo);

  // Insert rank
  current = current->findOrCreateChild(rank, 0, true);

  // Insert datatype
  current = current->findOrCreateChild(ArrayOptions::dataType(shapeInfo), 1, true);

  // Insert order
  current = current->findOrCreateChild(shape::order(shapeInfo), 2, true);

  // Insert shape values
  const LongType* shape = shape::shapeOf(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(shape[i], 3 + i, true);
  }

  // Insert stride values
  const LongType* strides = shape::stride(shapeInfo);
  for (int i = 0; i < rank; i++) {
    current = current->findOrCreateChild(strides[i], 3 + rank + i, false);
  }

  if (!current->buffer()) {
    auto buffer = createBuffer(shapeInfo);
    current->setBuffer(buffer);
    updateThreadCache(shapeInfo, buffer);
#if defined(SD_GCC_FUNCTRACE)
    current->collectStoreStackTrace();
#endif
  }

  return current->buffer();
}

ConstantShapeBuffer* DirectShapeTrie::getOrCreate(const LongType* shapeInfo) {
  validateShapeInfo(shapeInfo);

  // Check thread-local cache first
  for (const auto& entry : _threadCache.entries) {
    if (shapeInfoEqual(entry.first, shapeInfo)) {
      return entry.second;
    }
  }

  size_t stripeIdx = getStripeIndex(shapeInfo);

  // Try read-only search first
  {
    std::shared_lock<std::shared_mutex> readLock(_mutexes[stripeIdx]);
    if (auto buffer = search(shapeInfo, stripeIdx)) {
      updateThreadCache(shapeInfo, buffer);
      return buffer;
    }
  }

  // If not found, acquire write lock and try again
  {
    std::unique_lock<std::shared_mutex> writeLock(_mutexes[stripeIdx]);
    if (auto buffer = search(shapeInfo, stripeIdx)) {  // Double-check
      updateThreadCache(shapeInfo, buffer);
      return buffer;
    }
    return insert(shapeInfo, stripeIdx);
  }
}

bool DirectShapeTrie::exists(const LongType* shapeInfo) const {
  validateShapeInfo(shapeInfo);

  for (const auto& entry : _threadCache.entries) {
    if (shapeInfoEqual(entry.first, shapeInfo)) {
      return true;
    }
  }

  size_t stripeIdx = getStripeIndex(shapeInfo);
  std::shared_lock<std::shared_mutex> readLock(_mutexes[stripeIdx]);
  return search(shapeInfo, stripeIdx) != nullptr;
}

} // namespace sd