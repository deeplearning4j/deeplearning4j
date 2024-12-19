/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_MODULARHASHER_H
#define LIBND4J_MODULARHASHER_H

#include <vector>
#include <cstdint>
#include <initializer_list>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

namespace sd {
namespace helpers {
namespace detail {

// Common constants
extern const uint64_t GOLDEN_RATIO;
extern const uint64_t INITIAL_HASH;

// Base template for SIMD operations
template<typename T>
struct SIMDHasher {
    static uint64_t hash_chunk(const T* data, size_t size, uint64_t initial_hash);
};

// Specializations declared for uint64_t
template<>
struct SIMDHasher<uint64_t> {
    static uint64_t hash_chunk(const uint64_t* data, size_t size, uint64_t initial_hash);
};

// Template for handling different types of data chunks
template<typename T>
class DataChunkHasher {
public:
    static uint64_t hash_data(const T* data, size_t size, uint64_t initial_hash = INITIAL_HASH);
};

// Specialization for floating-point values declared
template<>
class DataChunkHasher<double> {
public:
    static uint64_t hash_data(const double* data, size_t size, uint64_t initial_hash = INITIAL_HASH);
};

// Main hasher class
class ModularHasher {
public:
    template<typename T>
    static uint64_t hash_vector(const std::vector<T>& vec, uint64_t initial_hash = INITIAL_HASH);
    
    static uint64_t combine_hashes(std::initializer_list<uint64_t> hashes);
    
    static uint64_t hash_scalar(uint64_t value, uint64_t initial_hash = INITIAL_HASH);
};

} // namespace detail
} // namespace helpers
} // namespace sd

#endif //LIBND4J_MODULARHASHER_H