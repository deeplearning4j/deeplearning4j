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

#ifndef LIBND4J_DIRECTSHAPETRIE_H
#define LIBND4J_DIRECTSHAPETRIE_H

#include <system/common.h>

#include <array>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "exceptions/backward.hpp"

namespace sd {

class ConstantShapeBuffer;

class SD_LIB_EXPORT ShapeTrieNode {
private:
    std::vector<std::unique_ptr<ShapeTrieNode>> _children;
    std::atomic<ConstantShapeBuffer*> _buffer;
    LongType _value;
    int _level;
    bool _isShape;
#if defined(SD_GCC_FUNCTRACE)
    backward::StackTrace st;
    backward::StackTrace storeStackTrace;
#endif

public:
    ShapeTrieNode(LongType value = 0, int level = 0, bool isShape = true);
    ~ShapeTrieNode();

    ShapeTrieNode* findOrCreateChild(LongType value, int level, bool isShape);
    const std::vector<std::unique_ptr<ShapeTrieNode>>& children() const;
    ConstantShapeBuffer* buffer() const;
    void setBuffer(ConstantShapeBuffer* buf);
    LongType value() const;
    int level() const;
    bool isShape() const;
    void collectStoreStackTrace();
};

class SD_LIB_EXPORT DirectShapeTrie {
private:
    static const size_t NUM_STRIPES = 32;
    std::array<std::unique_ptr<ShapeTrieNode>, NUM_STRIPES> _roots;
    mutable std::array<std::shared_mutex, NUM_STRIPES> _mutexes;  // Marked mutable for const member functions

    struct ThreadCache {
        static const size_t CACHE_SIZE = 1024;
        std::vector<std::pair<const LongType*, ConstantShapeBuffer*>> entries;
        ThreadCache();
    };
    
    static thread_local ThreadCache _threadCache;

    size_t computeHash(const LongType* shapeInfo) const;
    size_t getStripeIndex(const LongType* shapeInfo) const;
    bool shapeInfoEqual(const LongType* a, const LongType* b) const;
    void updateThreadCache(const LongType* shapeInfo, ConstantShapeBuffer* buffer);
    ConstantShapeBuffer* createBuffer(const LongType* shapeInfo);
    void validateShapeInfo(const LongType* shapeInfo) const;
    ConstantShapeBuffer* insert(const LongType* shapeInfo, size_t stripeIdx);
    ConstantShapeBuffer* search(const LongType* shapeInfo, size_t stripeIdx) const;
    const ShapeTrieNode* findChild(const ShapeTrieNode* node, LongType value, 
                                  int level, bool isShape) const;

public:
    DirectShapeTrie();
    ConstantShapeBuffer* getOrCreate(const LongType* shapeInfo);
    bool exists(const LongType* shapeInfo) const;
};

}  // namespace sd

#endif //LIBND4J_DIRECTSHAPETRIE_H