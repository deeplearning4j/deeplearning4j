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

#ifndef LIBND4J_GRAPH_RNG_H
#define LIBND4J_GRAPH_RNG_H

#include <array/DataTypeUtils.h>
#include <helpers/logger.h>
#include <math/templatemath.h>
#include <system/op_boilerplate.h>
#include <types/u32.h>
#include <types/u64.h>
#include <chrono>
#include <stdexcept>

namespace sd {
namespace graph {

#if defined(__CUDACC__)
#ifndef __JAVACPP_HACK__

class SD_LIB_EXPORT CudaManagedRandomGenerator {
 private:
 protected:
  void *devHolder;

 public:
  void *operator new(size_t len) {
    void *ptr;
    auto res = cudaHostAlloc(&ptr, len, cudaHostAllocDefault);
    if (res != 0) THROW_EXCEPTION("CudaManagedRandomGenerator: failed to allocate memory");
    return ptr;
  }

  void operator delete(void *ptr) { cudaFreeHost(ptr); }
};

class SD_LIB_EXPORT RandomGenerator : public CudaManagedRandomGenerator {
 private:
  u64 _rootState;
  u64 _nodeState;

  static SD_INLINE LongType currentMilliseconds();

 public:
  SD_INLINE RandomGenerator(LongType rootSeed = 0, LongType nodeSeed = 0);
  SD_INLINE SD_HOST void setStates(LongType rootSeed, LongType nodeState = 0);

  template <typename T>
  SD_INLINE SD_HOST_DEVICE T relativeT(LongType index, T from, T to);

  template <typename T>
  SD_INLINE SD_HOST_DEVICE T relativeT(LongType index);

  SD_INLINE SD_HOST_DEVICE int relativeInt(LongType index);
  SD_INLINE SD_HOST_DEVICE LongType relativeLong(LongType index);
  SD_INLINE SD_HOST_DEVICE void rewindH(uint64_t steps);

  SD_INLINE SD_HOST void setSeed(int seed) { _nodeState._ulong = static_cast<uint64_t>(seed); }
  SD_INLINE SD_HOST void setSeed(uint64_t seed) { _nodeState._ulong = seed; }

  SD_INLINE SD_HOST_DEVICE LongType rootState() { return _rootState._long; }
  SD_INLINE SD_HOST_DEVICE LongType nodeState() { return _nodeState._long; }

  SD_INLINE SD_HOST_DEVICE uint32_t xoroshiro32(uint64_t index);
  SD_INLINE SD_HOST_DEVICE uint64_t xoroshiro64(uint64_t index);
};
#endif

#else

class SD_LIB_EXPORT RandomGenerator {
 private:
  u64 _rootState;
  u64 _nodeState;

  static SD_INLINE LongType currentMilliseconds();

 public:
  SD_INLINE RandomGenerator(LongType rootSeed = 0, LongType nodeSeed = 0);
  SD_INLINE SD_HOST void setStates(LongType rootSeed, LongType nodeState = 0);

  template <typename T>
  SD_INLINE SD_HOST_DEVICE T relativeT(LongType index, T from, T to);

  template <typename T>
  SD_INLINE SD_HOST_DEVICE T relativeT(LongType index);

  SD_INLINE SD_HOST_DEVICE int relativeInt(LongType index);
  SD_INLINE SD_HOST_DEVICE LongType relativeLong(LongType index);
  SD_INLINE SD_HOST_DEVICE void rewindH(uint64_t steps);

  SD_INLINE SD_HOST void setSeed(int seed) { _nodeState._ulong = static_cast<uint64_t>(seed); }
  SD_INLINE SD_HOST void setSeed(uint64_t seed) { _nodeState._ulong = seed; }

  SD_INLINE SD_HOST_DEVICE LongType rootState() { return _rootState._long; }
  SD_INLINE SD_HOST_DEVICE LongType nodeState() { return _nodeState._long; }

  SD_INLINE SD_HOST_DEVICE uint32_t xoroshiro32(uint64_t index);
  SD_INLINE SD_HOST_DEVICE uint64_t xoroshiro64(uint64_t index);
};

#endif

// Implementation of member functions (common for both CUDA and non-CUDA versions)

SD_INLINE RandomGenerator::RandomGenerator(LongType rootSeed, LongType nodeSeed) {
  _rootState._long = (rootSeed == 0) ? currentMilliseconds() : rootSeed;
  _nodeState._long = (nodeSeed != 0) ? nodeSeed : 1298567341LL;
}

SD_INLINE void RandomGenerator::setStates(LongType rootSeed, LongType nodeSeed) {
  _rootState._long = (rootSeed == 0) ? currentMilliseconds() : rootSeed;
  _nodeState._long = (nodeSeed != 0) ? nodeSeed : 1298567341LL;
}

SD_INLINE LongType RandomGenerator::currentMilliseconds() {
  auto s = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(s).count();
}

// Template specializations for relativeT

#ifdef HAS_FLOAT32
template <>
SD_INLINE SD_HOST_DEVICE float RandomGenerator::relativeT<float>(LongType index) {
  u32 u;
  u._u32 = (0x3f800000 | (this->xoroshiro32(index) >> 9));
  return u._f32 - 1.0f;
}
#endif

#ifdef HAS_DOUBLE
template <>
SD_INLINE SD_HOST_DEVICE double RandomGenerator::relativeT<double>(LongType index) {
#ifdef __DOUBLE_RNG__
  u64 u;
  u._ulong = ((UINT64_C(0x3FF) << 52) | (this->xoroshiro64(index) >> 12));
  return u._double - 1.0;
#else
  return (double)relativeT<float>(index);
#endif
}
#endif

// Use unsigned long instead of uint64_t to avoid redefinition issues
#ifdef HAS_UINT64
template <>
SD_INLINE SD_HOST_DEVICE unsigned long RandomGenerator::relativeT<unsigned long>(LongType index) {
  return this->xoroshiro64(index);
}
#endif

#ifdef HAS_UINT32
template <>
SD_INLINE SD_HOST_DEVICE uint32_t RandomGenerator::relativeT<uint32_t>(LongType index) {
  return this->xoroshiro32(index);
}
#endif

#ifdef HAS_INT32
template <>
SD_INLINE SD_HOST_DEVICE int RandomGenerator::relativeT<int>(LongType index) {
  auto r = static_cast<int>(relativeT<uint32_t>(index));
  return r <= DataTypeUtils::max<int>() ? r : r % DataTypeUtils::max<int>();
}
#endif

#ifdef HAS_INT64
template <>
SD_INLINE SD_HOST_DEVICE LongType RandomGenerator::relativeT<LongType>(LongType index) {
  auto r = static_cast<sd::LongType>(relativeT<unsigned long>(index));
  return r <= DataTypeUtils::max<LongType>() ? r : r % DataTypeUtils::max<LongType>();
}
#endif

// Additional template specializations for integer types with single parameter
#ifdef HAS_INT8
template <>
SD_INLINE SD_HOST_DEVICE int8_t RandomGenerator::relativeT<int8_t>(LongType index) {
  // Return a value between 0 and max for int8_t
  float t = this->relativeT<float>(index);
  return static_cast<int8_t>(t * DataTypeUtils::max<int8_t>());
}
#endif

#ifdef HAS_UINT8
template <>
SD_INLINE SD_HOST_DEVICE uint8_t RandomGenerator::relativeT<uint8_t>(LongType index) {
  float t = this->relativeT<float>(index);
  return static_cast<uint8_t>(t * DataTypeUtils::max<uint8_t>());
}
#endif

#ifdef HAS_INT16
template <>
SD_INLINE SD_HOST_DEVICE int16_t RandomGenerator::relativeT<int16_t>(LongType index) {
  float t = this->relativeT<float>(index);
  return static_cast<int16_t>(t * DataTypeUtils::max<int16_t>());
}
#endif

#ifdef HAS_UINT16
template <>
SD_INLINE SD_HOST_DEVICE uint16_t RandomGenerator::relativeT<uint16_t>(LongType index) {
  float t = this->relativeT<float>(index);
  return static_cast<uint16_t>(t * DataTypeUtils::max<uint16_t>());
}
#endif

#ifdef HAS_BOOL
template <>
SD_INLINE SD_HOST_DEVICE bool RandomGenerator::relativeT<bool>(LongType index) {
  float t = this->relativeT<float>(index);
  return t > 0.5f;
}
#endif

#ifdef HAS_FLOAT16
template <>
SD_INLINE SD_HOST_DEVICE float16 RandomGenerator::relativeT<float16>(LongType index) {
  return static_cast<float16>(relativeT<float>(index));
}
#endif

#ifdef HAS_BFLOAT16
template <>
SD_INLINE SD_HOST_DEVICE bfloat16 RandomGenerator::relativeT<bfloat16>(LongType index) {
  return static_cast<bfloat16>(relativeT<float>(index));
}
#endif

// Generic template for relativeT with range parameters
template <typename T>
SD_INLINE SD_HOST_DEVICE T RandomGenerator::relativeT(LongType index, T from, T to) {
  auto t = this->relativeT<T>(index);
  return from + T(t * (to - from));
}

// Specializations for relativeT with range parameters
#ifdef HAS_INT64
template <>
SD_INLINE SD_HOST_DEVICE LongType RandomGenerator::relativeT(LongType index, LongType from, LongType to) {
  auto t = this->relativeT<double>(index);
  return from + LongType(t * (to - from));
}
#endif

#ifdef HAS_INT32
template <>
SD_INLINE SD_HOST_DEVICE int RandomGenerator::relativeT(LongType index, int from, int to) {
  auto t = this->relativeT<float>(index);
  return from + int(t * (to - from));
}
#endif

// Template specializations for integer types with range parameters
#ifdef HAS_INT8
template <>
SD_INLINE SD_HOST_DEVICE int8_t RandomGenerator::relativeT(LongType index, int8_t from, int8_t to) {
  // Use float for intermediate calculation to get proper distribution
  float t = this->relativeT<float>(index);
  return from + static_cast<int8_t>(t * (to - from));
}
#endif

#ifdef HAS_UINT8
template <>
SD_INLINE SD_HOST_DEVICE uint8_t RandomGenerator::relativeT(LongType index, uint8_t from, uint8_t to) {
  float t = this->relativeT<float>(index);
  return from + static_cast<uint8_t>(t * (to - from));
}
#endif

#ifdef HAS_INT16
template <>
SD_INLINE SD_HOST_DEVICE int16_t RandomGenerator::relativeT(LongType index, int16_t from, int16_t to) {
  float t = this->relativeT<float>(index);
  return from + static_cast<int16_t>(t * (to - from));
}
#endif

#ifdef HAS_UINT16
template <>
SD_INLINE SD_HOST_DEVICE uint16_t RandomGenerator::relativeT(LongType index, uint16_t from, uint16_t to) {
  float t = this->relativeT<float>(index);
  return from + static_cast<uint16_t>(t * (to - from));
}
#endif

#ifdef HAS_BOOL
template <>
SD_INLINE SD_HOST_DEVICE bool RandomGenerator::relativeT(LongType index, bool from, bool to) {
  float t = this->relativeT<float>(index);
  return t > 0.5f ? to : from;
}
#endif

#ifdef HAS_FLOAT16
template <>
SD_INLINE SD_HOST_DEVICE float16 RandomGenerator::relativeT(LongType index, float16 from, float16 to) {
  float t = this->relativeT<float>(index);
  return from + static_cast<float16>(t * (to - from));
}
#endif

#ifdef HAS_BFLOAT16
template <>
SD_INLINE SD_HOST_DEVICE bfloat16 RandomGenerator::relativeT(LongType index, bfloat16 from, bfloat16 to) {
  float t = this->relativeT<float>(index);
  return from + static_cast<bfloat16>(t * (to - from));
}
#endif

#ifdef HAS_FLOAT32
template <>
SD_INLINE SD_HOST_DEVICE float RandomGenerator::relativeT(LongType index, float from, float to) {
  auto t = this->relativeT<float>(index);
  return from + (t * (to - from));
}
#endif

#ifdef HAS_DOUBLE
template <>
SD_INLINE SD_HOST_DEVICE double RandomGenerator::relativeT(LongType index, double from, double to) {
  auto t = this->relativeT<double>(index);
  return from + (t * (to - from));
}
#endif

// Generic fallback template - only compiled if no specialization exists
template <typename T>
SD_INLINE SD_HOST_DEVICE T RandomGenerator::relativeT(LongType index) {
  return static_cast<T>(relativeT<float>(index));
}

SD_INLINE SD_HOST_DEVICE int RandomGenerator::relativeInt(LongType index) {
#ifdef HAS_UINT32
  auto r = static_cast<int>(relativeT<uint32_t>(index));
  return r <= DataTypeUtils::max<int>() ? r : r % DataTypeUtils::max<int>();
#else
  return 0;  // Fallback if no uint32_t
#endif
}

SD_INLINE SD_HOST_DEVICE LongType RandomGenerator::relativeLong(LongType index) {
#ifdef HAS_UINT64
  auto r = static_cast<LongType>(relativeT<unsigned long>(index));
  return r <= DataTypeUtils::max<LongType>() ? r : r % DataTypeUtils::max<LongType>();
#else
  return 0;  // Fallback if no uint64_t
#endif
}

// Helper functions
static SD_INLINE SD_HOST_DEVICE uint32_t rotl(const uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
static SD_INLINE SD_HOST_DEVICE uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

SD_INLINE SD_HOST_DEVICE uint32_t RandomGenerator::xoroshiro32(uint64_t index) {
  auto s0 = _rootState._ulong;
  auto s1 = _nodeState._ulong;

  s0 |= ((index + 2) * (s1 + 24243287));
  s1 ^= ((index + 2) * (s0 + 723829));

  unsigned long val = s1 ^ s0;
  int *pHalf = reinterpret_cast<int *>(&val);

  return rotl(*pHalf * 0x9E3779BB, 5) * 5;
}

SD_INLINE SD_HOST_DEVICE uint64_t RandomGenerator::xoroshiro64(uint64_t index) {
  uint64_t upper = ((uint64_t)xoroshiro32(index)) << 32;
  // Use direct bit manipulation instead of sd_rotl to avoid template issues
  uint64_t rotated = (index << 32) | (index >> 32);
  uint32_t lower = xoroshiro32(rotated);
  return upper + lower;
}

SD_INLINE SD_HOST_DEVICE void RandomGenerator::rewindH(uint64_t steps) {
  auto s0 = _nodeState._du32._v0;
  auto s1 = _nodeState._du32._v1;

  s1 ^= s0;
  _nodeState._du32._v0 = rotl(s0, 26) ^ s1 ^ (s1 << 9);
  _nodeState._du32._v1 = rotl(s1, 13);

  _nodeState._long ^= (steps ^ 0xdeadbeef);
}

}  // namespace graph
}  // namespace sd


#endif // LIBND4J_GRAPH_RNG_H