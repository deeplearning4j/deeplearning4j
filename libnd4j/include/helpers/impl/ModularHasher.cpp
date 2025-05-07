#include <helpers/ModularHasher.h>
#include <cstring>

namespace sd {
namespace helpers {
namespace detail {

const uint64_t GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
const uint64_t INITIAL_HASH = 14695981039346656037ULL;

// Specialization for uint64_t
template<> uint64_t SIMDHasher<uint64_t>::hash_chunk(const uint64_t* data, size_t size, uint64_t initial_hash) {
  uint64_t hash = initial_hash;

#if defined(__ARM_NEON)
  uint64x2_t hash_vec = vdupq_n_u64(initial_hash);
  const uint64x2_t golden = vdupq_n_u64(GOLDEN_RATIO);

  for (size_t i = 0; i < size - 1; i += 2) {
    uint64x2_t val = vld1q_u64(data + i);
    hash_vec = veorq_u64(hash_vec, val);
    // Extract lower 32 bits of each 64-bit lane
    uint32x2_t low_hash = vmovn_u64(hash_vec);
    uint32x2_t low_golden = vmovn_u64(golden);
    // Perform 32x32 -> 64 bit widening multiply
    hash_vec = vmull_u32(low_hash, low_golden);
  }

  uint64_t tmp[2];
  vst1q_u64(tmp, hash_vec);
  hash = tmp[0] ^ tmp[1];

#elif defined(__AVX2__)
  __m256i hash_vec = _mm256_set1_epi64x(initial_hash);
  const __m256i golden_vec = _mm256_set1_epi64x(GOLDEN_RATIO);

  for (size_t i = 0; i < size - 3; i += 4) {
    __m256i val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
    hash_vec = _mm256_xor_si256(hash_vec, val);
    hash_vec = _mm256_mul_epi32(hash_vec, golden_vec);
  }

  uint64_t tmp[4];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), hash_vec);
  hash = tmp[0] ^ tmp[1] ^ tmp[2] ^ tmp[3];

#elif defined(__SSE4_2__)
  __m128i hash_vec = _mm_set1_epi64x(initial_hash);
  const __m128i golden_vec = _mm_set1_epi64x(GOLDEN_RATIO);

  for (size_t i = 0; i < size - 1; i += 2) {
    __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
    hash_vec = _mm_xor_si128(hash_vec, val);
    hash_vec = _mm_mul_epi32(hash_vec, golden_vec);
  }

  uint64_t tmp[2];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), hash_vec);
  hash = tmp[0] ^ tmp[1];

#else
  if(size >= 4) {
    // Scalar fallback with unrolling
    for (size_t i = 0; i < size - 3; i += 4) {
      hash ^= data[i];
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
      hash ^= data[i+1];
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
      hash ^= data[i+2];
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
      hash ^= data[i+3];
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
  }

#endif

  // Handle remaining elements
  size_t remainder = size % 4;
  if(size >= 4) {
    size_t start = size - remainder;
    for (size_t i = start; i < size; i++) {
      hash ^= data[i];
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
  }
  return hash;
}

// Specialization for double
uint64_t DataChunkHasher<double>::hash_data(const double* data, size_t size, uint64_t initial_hash) {
  return SIMDHasher<uint64_t>::hash_chunk(
      reinterpret_cast<const uint64_t*>(data),
      size,
      initial_hash
  );
}

uint64_t ModularHasher::combine_hashes(std::initializer_list<uint64_t> hashes) {
  uint64_t result = INITIAL_HASH;
  for (uint64_t h : hashes) {
    result ^= h;
    result = (result * GOLDEN_RATIO) ^ (result >> 32);
  }
  return result;
}

uint64_t ModularHasher::hash_scalar(uint64_t value, uint64_t initial_hash) {
  uint64_t hash = initial_hash;
  hash ^= value;
  return (hash * GOLDEN_RATIO) ^ (hash >> 32);
}

// Explicit template instantiations
template uint64_t ModularHasher::hash_vector<uint64_t>(const std::vector<uint64_t>&, uint64_t);
template uint64_t ModularHasher::hash_vector<double>(const std::vector<double>&, uint64_t);
template uint64_t ModularHasher::hash_vector<int64_t>(const std::vector<int64_t>&, uint64_t);

} // namespace detail
} // namespace helpers
} // namespace sd