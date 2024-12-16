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
//  @author raver119@gmail.com
//

#include "../TadDescriptor.h"

#include <algorithm>

namespace sd {
// already defined for NEC compiler
TadDescriptor::TadDescriptor(const TadDescriptor &other) {
  _originalShape = other._originalShape;
  _axis = other._axis;
  _unitiesInShape = other._unitiesInShape;
}

TadDescriptor::TadDescriptor(const LongType *originalShape, const LongType *dimensions, const LongType length,
                             const bool keepUnitiesInShape) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(originalShape, false);

  _axis.resize(length);
  for (LongType e = 0; e < length; e++) {
    _axis[e] = dimensions[e];
  }

  if (length > 1) std::sort(_axis.begin(), _axis.end());

  _originalShape = *descriptor;
  _unitiesInShape = keepUnitiesInShape;
}

TadDescriptor::TadDescriptor(const ShapeDescriptor &descriptor, const std::vector<LongType> &dimensions,
                             const bool keepUnitiesInShape) {
  _originalShape = descriptor;
  _axis = dimensions;
  _unitiesInShape = keepUnitiesInShape;

  if (_axis.size() > 1) std::sort(_axis.begin(), _axis.end());
}

bool TadDescriptor::operator==(const TadDescriptor &other) const {
  return std::tie(_originalShape, _axis, _unitiesInShape) ==
         std::tie(other._originalShape, other._axis, other._unitiesInShape);
}

bool TadDescriptor::operator<(const TadDescriptor &other) const {
  return std::tie(_originalShape, _axis, _unitiesInShape) <
         std::tie(other._originalShape, other._axis, other._unitiesInShape);
}

std::vector<LongType> &TadDescriptor::axis() { return _axis; }

ShapeDescriptor &TadDescriptor::originalShape() { return _originalShape; }

ShapeDescriptor const &TadDescriptor::originalShapeConst() const { return _originalShape; }

bool TadDescriptor::areUnitiesinShape() const { return _unitiesInShape; }
}  // namespace sd

namespace std {
size_t hash<sd::TadDescriptor>::operator()(const sd::TadDescriptor &k) const {
  constexpr uint64_t GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
  uint64_t hash = 14695981039346656037ULL;  // FNV offset basis

  // Add flag for unities in shape
  hash ^= static_cast<uint64_t>(k.areUnitiesinShape());
  hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);

  // Combine with original shape hash
  hash ^= std::hash<sd::ShapeDescriptor>()(k.originalShapeConst());
  hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);

  // Process axes efficiently
  auto axes = const_cast<sd::TadDescriptor &>(k).axis();
  const size_t num_axes = axes.size();
  const auto* axis_data = axes.data();

#if defined(__ARM_NEON)
  if (num_axes >= 2) {
    const uint64x2_t golden = vdupq_n_u64(GOLDEN_RATIO);
    uint64x2_t hash_vec = vdupq_n_u64(hash);

    // Process 2 axes at a time
    for (size_t i = 0; i < num_axes - 1; i += 2) {
      uint64x2_t data = vcombine_u64(
          vcreate_u64(static_cast<uint64_t>(axis_data[i])),
          vcreate_u64(static_cast<uint64_t>(axis_data[i + 1]))
      );
      hash_vec = veorq_u64(hash_vec, data);
      hash_vec = vmulq_u64(hash_vec, golden);
    }

    uint64_t hash_array[2];
    vst1q_u64(hash_array, hash_vec);
    hash = hash_array[0] ^ hash_array[1];

    // Handle remaining axis
    if (num_axes & 1) {
      hash ^= static_cast<uint64_t>(axis_data[num_axes - 1]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
  }
#elif defined(__AVX2__)
  if (num_axes >= 4) {
    const __m256i golden_vec = _mm256_set1_epi64x(GOLDEN_RATIO);
    __m256i hash_vec = _mm256_set1_epi64x(hash);

    // Process 4 axes at a time
    for (size_t i = 0; i < num_axes - 3; i += 4) {
      __m256i data = _mm256_set_epi64x(
          static_cast<int64_t>(axis_data[i + 3]),
          static_cast<int64_t>(axis_data[i + 2]),
          static_cast<int64_t>(axis_data[i + 1]),
          static_cast<int64_t>(axis_data[i])
      );
      hash_vec = _mm256_xor_si256(hash_vec, data);
      hash_vec = _mm256_mul_epi32(hash_vec, golden_vec);
    }

    uint64_t hash_array[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(hash_array), hash_vec);
    hash = hash_array[0] ^ hash_array[1] ^ hash_array[2] ^ hash_array[3];

    // Handle remaining axes
    for (size_t i = (num_axes / 4) * 4; i < num_axes; i++) {
      hash ^= static_cast<uint64_t>(axis_data[i]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
  }
#elif defined(__SSE4_2__)
  if (num_axes >= 2) {
    const __m128i golden_vec = _mm_set1_epi64x(GOLDEN_RATIO);
    __m128i hash_vec = _mm_set1_epi64x(hash);

    // Process 2 axes at a time
    for (size_t i = 0; i < num_axes - 1; i += 2) {
      __m128i data = _mm_set_epi64x(
          static_cast<int64_t>(axis_data[i + 1]),
          static_cast<int64_t>(axis_data[i])
      );
      hash_vec = _mm_xor_si128(hash_vec, data);
      hash_vec = _mm_mul_epi32(hash_vec, golden_vec);
    }

    uint64_t hash_array[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(hash_array), hash_vec);
    hash = hash_array[0] ^ hash_array[1];

    // Handle remaining axis
    if (num_axes & 1) {
      hash ^= static_cast<uint64_t>(axis_data[num_axes - 1]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
  }
#else
  // Scalar fallback with unrolling for small arrays
  if (num_axes >= 4) {
    for (size_t i = 0; i < num_axes - 3; i += 4) {
      hash ^= static_cast<uint64_t>(axis_data[i]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
      hash ^= static_cast<uint64_t>(axis_data[i + 1]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
      hash ^= static_cast<uint64_t>(axis_data[i + 2]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
      hash ^= static_cast<uint64_t>(axis_data[i + 3]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
  }
#endif

// Handle remaining elements for small arrays or when SIMD wasn't used
#if (defined(__ARM_NEON) || defined(__AVX2__) || defined(__SSE4_2__))
  if (num_axes < 2) {  // For very small arrays or remaining elements
#endif
    for (size_t i = 0; i < num_axes; i++) {
      hash ^= static_cast<uint64_t>(axis_data[i]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
#if (defined(__ARM_NEON) || defined(__AVX2__) || defined(__SSE4_2__))
  }
#endif

  return hash;
}
}  // namespace std
