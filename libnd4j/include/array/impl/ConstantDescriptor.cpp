/* ******************************************************************************
 *
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

//
// @author raver119@gmail.com
//
#include <array/ConstantDescriptor.h>
#include <array/DataTypeUtils.h>

#include <stdexcept>

namespace sd {
ConstantDescriptor::ConstantDescriptor(double *values, int length) {
  for (int e = 0; e < length; e++) _floatValues.emplace_back(values[e]);
}

ConstantDescriptor::ConstantDescriptor(LongType const *values, int length) {
  for (int e = 0; e < length; e++) _integerValues.emplace_back(values[e]);
}

ConstantDescriptor::ConstantDescriptor(std::initializer_list<double> values) { _floatValues = values; }

ConstantDescriptor::ConstantDescriptor(std::vector<LongType> &values) { _integerValues = values; }

ConstantDescriptor::ConstantDescriptor(std::vector<double> &values) { _floatValues = values; }

// equal to operator
bool ConstantDescriptor::operator==(const ConstantDescriptor &other) const {
  return std::tie(_floatValues, _integerValues) == std::tie(other._floatValues, other._integerValues);
}

// less than operator
bool ConstantDescriptor::operator<(const ConstantDescriptor &other) const {
  return std::tie(_floatValues, _integerValues) < std::tie(other._floatValues, other._integerValues);
}

bool ConstantDescriptor::isInteger() const { return !_integerValues.empty(); }

bool ConstantDescriptor::isFloat() const { return !_floatValues.empty(); }

const std::vector<LongType> &ConstantDescriptor::integerValues() const { return _integerValues; }

const std::vector<double> &ConstantDescriptor::floatValues() const { return _floatValues; }

LongType ConstantDescriptor::length() const {
  return isInteger() ? _integerValues.size() : isFloat() ? _floatValues.size() : 0L;
}
}  // namespace sd

namespace std {
size_t hash<sd::ConstantDescriptor>::operator()(const sd::ConstantDescriptor &k) const {
  constexpr uint64_t GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
  uint64_t hash = 14695981039346656037ULL;

  hash ^= k.isInteger();
  hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);

  if (k.isInteger()) {
    const auto& values = k.integerValues();
    const size_t size = values.size();
    const auto* data = values.data();

#if defined(__ARM_NEON)
    uint64x2_t hash_vec = vdupq_n_u64(hash);
    const uint64x2_t golden = vdupq_n_u64(GOLDEN_RATIO);

    for (size_t i = 0; i < size - 1; i += 2) {
      uint64x2_t val = vld1q_u64(reinterpret_cast<const uint64_t*>(data + i));
      hash_vec = veorq_u64(hash_vec, val);
      hash_vec = vmulq_u64(hash_vec, golden);
    }

    uint64_t tmp[2];
    vst1q_u64(tmp, hash_vec);
    hash = tmp[0] ^ tmp[1];
#elif defined(__AVX2__)
    const __m256i golden_vec = _mm256_set1_epi64x(GOLDEN_RATIO);
    __m256i hash_vec = _mm256_set1_epi64x(hash);

    for (size_t i = 0; i < size - 3; i += 4) {
      __m256i val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
      hash_vec = _mm256_xor_si256(hash_vec, val);
      hash_vec = _mm256_mul_epi32(hash_vec, golden_vec);
    }

    uint64_t tmp[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), hash_vec);
    hash = tmp[0] ^ tmp[1] ^ tmp[2] ^ tmp[3];
#elif defined(__SSE4_2__)
    const __m128i golden_vec = _mm_set1_epi64x(GOLDEN_RATIO);
    __m128i hash_vec = _mm_set1_epi64x(hash);

    for (size_t i = 0; i < size - 1; i += 2) {
      __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
      hash_vec = _mm_xor_si128(hash_vec, val);
      hash_vec = _mm_mul_epi32(hash_vec, golden_vec);
    }

    uint64_t tmp[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), hash_vec);
    hash = tmp[0] ^ tmp[1];
#else
    for (size_t i = 0; i < size; i++) {
      hash ^= data[i];
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
#endif

// Handle remaining elements
#if defined(__ARM_NEON) || defined(__AVX2__) || defined(__SSE4_2__)
    size_t start = ((size / 4) * 4);
    for (size_t i = start; i < size; i++) {
      hash ^= data[i];
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
#endif
  } else {
    const auto& values = k.floatValues();
    const size_t size = values.size();
    const auto* data = values.data();

#if defined(__ARM_NEON)
    uint64x2_t hash_vec = vdupq_n_u64(hash);
    const uint64x2_t golden = vdupq_n_u64(GOLDEN_RATIO);

    for (size_t i = 0; i < size - 1; i += 2) {
      float64x2_t val = vld1q_f64(data + i);
      uint64x2_t bits = vreinterpretq_u64_f64(val);
      hash_vec = veorq_u64(hash_vec, bits);
      hash_vec = vmulq_u64(hash_vec, golden);
    }

    uint64_t tmp[2];
    vst1q_u64(tmp, hash_vec);
    hash = tmp[0] ^ tmp[1];
#elif defined(__AVX2__)
    const __m256i golden_vec = _mm256_set1_epi64x(GOLDEN_RATIO);
    __m256i hash_vec = _mm256_set1_epi64x(hash);

    for (size_t i = 0; i < size - 3; i += 4) {
      __m256d val = _mm256_loadu_pd(data + i);
      __m256i bits = _mm256_castpd_si256(val);
      hash_vec = _mm256_xor_si256(hash_vec, bits);
      hash_vec = _mm256_mul_epi32(hash_vec, golden_vec);
    }

    uint64_t tmp[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), hash_vec);
    hash = tmp[0] ^ tmp[1] ^ tmp[2] ^ tmp[3];
#elif defined(__SSE4_2__)
    const __m128i golden_vec = _mm_set1_epi64x(GOLDEN_RATIO);
    __m128i hash_vec = _mm_set1_epi64x(hash);

    for (size_t i = 0; i < size - 1; i += 2) {
      __m128d val = _mm_loadu_pd(data + i);
      __m128i bits = _mm_castpd_si128(val);
      hash_vec = _mm_xor_si128(hash_vec, bits);
      hash_vec = _mm_mul_epi32(hash_vec, golden_vec);
    }

    uint64_t tmp[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), hash_vec);
    hash = tmp[0] ^ tmp[1];
#else
    for (size_t i = 0; i < size; i++) {
      uint64_t bits;
      memcpy(&bits, &data[i], sizeof(double));
      hash ^= bits;
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
#endif

// Handle remaining elements
#if defined(__ARM_NEON) || defined(__AVX2__) || defined(__SSE4_2__)
    size_t start = ((size / 4) * 4);
    for (size_t i = start; i < size; i++) {
      uint64_t bits;
      memcpy(&bits, &data[i], sizeof(double));
      hash ^= bits;
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
#endif
  }

  return hash;
}
}  // namespace std
