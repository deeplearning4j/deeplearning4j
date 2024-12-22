#ifndef LIBND4J_MODULARHASHER_H
#define LIBND4J_MODULARHASHER_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

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
  static uint64_t hash_chunk(const T* data, size_t size, uint64_t initial_hash) {
    uint64_t hash = initial_hash;
    for (size_t i = 0; i < size; i++) {
      hash ^= static_cast<uint64_t>(data[i]);
      hash = (hash * GOLDEN_RATIO) ^ (hash >> 32);
    }
    return hash;
  }
};

// Template for handling different types of data chunks
template<typename T>
class DataChunkHasher {
 public:
  static uint64_t hash_data(const T* data, size_t size, uint64_t initial_hash = INITIAL_HASH) {
    return SIMDHasher<uint64_t>::hash_chunk(
        reinterpret_cast<const uint64_t*>(data),
        (size * sizeof(T) + 7) / 8,
        initial_hash
    );
  }
};

// Forward declare specializations

template<>
class DataChunkHasher<double> {
 public:
  static uint64_t hash_data(const double* data, size_t size, uint64_t initial_hash = INITIAL_HASH);
};

// Main hasher class
class ModularHasher {
 public:
  template<typename T>
  static uint64_t hash_vector(const std::vector<T>& vec, uint64_t initial_hash = INITIAL_HASH) {
    return DataChunkHasher<T>::hash_data(vec.data(), vec.size(), initial_hash);
  }

  static uint64_t combine_hashes(std::initializer_list<uint64_t> hashes);
  static uint64_t hash_scalar(uint64_t value, uint64_t initial_hash = INITIAL_HASH);
};

} // namespace detail
} // namespace helpers
} // namespace sd

#endif //LIBND4J_MODULARHASHER_H