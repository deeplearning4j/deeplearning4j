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
//  @author raver119@protonmail.com
//

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
#ifdef __CUDACC__
class SD_LIB_EXPORT CudaManagedRandomGenerator {
 private:
 protected:
  void *devHolder;

 public:
  void *operator new(size_t len) {
    void *ptr;
    auto res = cudaHostAlloc(&ptr, len, cudaHostAllocDefault);
    if (res != 0) throw std::runtime_error("CudaManagedRandomGenerator: failed to allocate memory");

    return ptr;
  }

  void operator delete(void *ptr) { cudaFreeHost(ptr); }
};

class SD_LIB_EXPORT RandomGenerator : public CudaManagedRandomGenerator {
#else
class SD_LIB_EXPORT RandomGenerator {
#endif
 private:
#ifndef __CUDACC__
  void *placeHolder;
#endif
  // GRAPH-LEVEL STATE
  u64 _rootState;

  // NODE-LEVEL STATE
  u64 _nodeState;

  /**
   * Utility method, returns number of milliseconds since 1970
   * Leave this static if possible to avoid problems in constructor
   */
  static SD_INLINE sd::LongType currentMilliseconds();

 public:
  SD_INLINE SD_HOST_DEVICE uint32_t xoroshiro32(uint64_t index);
  SD_INLINE SD_HOST_DEVICE uint64_t xoroshiro64(uint64_t index);

  /**
   * This method returns integer value between 0 and MAX_UINT
   */
  // uint32_t relativeUInt32(sd::LongType index);

 public:
  SD_INLINE RandomGenerator(sd::LongType rootSeed = 0, sd::LongType nodeSeed = 0);

  /**
   * This method allows to change graph-level state in runtime.
   * PLEASE NOTE: this method will change state of node as well.
   */
  SD_INLINE SD_HOST void setStates(sd::LongType rootSeed, sd::LongType nodeState = 0);

  /**
   * This method returns T value between from and to
   */
  template <typename T>
  SD_INLINE SD_HOST_DEVICE T relativeT(sd::LongType index, T from, T to);

  /**
   * This method returns T value between 0 and MAX_T
   */
  template <typename T>
  SD_INLINE SD_HOST_DEVICE T relativeT(sd::LongType index);

  /**
   * These two methods are made for JVM
   * @param index
   * @return
   */
  SD_INLINE SD_HOST_DEVICE int relativeInt(sd::LongType index);
  SD_INLINE SD_HOST_DEVICE sd::LongType relativeLong(sd::LongType index);

  SD_INLINE SD_HOST_DEVICE void rewindH(uint64_t steps);

  /**
   * These methods set up only node states, with non-changed root ones
   */
  SD_INLINE SD_HOST void setSeed(int seed) { _nodeState._ulong = static_cast<uint64_t>(seed); }

  SD_INLINE SD_HOST void setSeed(uint64_t seed) { _nodeState._ulong = seed; }

  SD_INLINE SD_HOST_DEVICE sd::LongType rootState() { return _rootState._long; }

  SD_INLINE SD_HOST_DEVICE sd::LongType nodeState() { return _nodeState._long; }
};

SD_INLINE RandomGenerator::RandomGenerator(sd::LongType rootSeed, sd::LongType nodeSeed) {
  // this seed is used graph-level state
  if (rootSeed == 0) rootSeed = currentMilliseconds();

  // graph-level state is just first seed
  _rootState._long = rootSeed;

  // used to build second, node state
  _nodeState._long = (nodeSeed != 0 ? nodeSeed : 1298567341LL);
}

SD_INLINE void RandomGenerator::setStates(sd::LongType rootSeed, sd::LongType nodeSeed) {
  // this seed is used graph-level state
  if (rootSeed == 0) rootSeed = currentMilliseconds();

  // graph-level state is just first seed
  _rootState._long = rootSeed;

  // used to build second, node state
  _nodeState._long = (nodeSeed != 0 ? nodeSeed : 1298567341LL);
}

SD_INLINE sd::LongType RandomGenerator::currentMilliseconds() {
  auto s = std::chrono::system_clock::now().time_since_epoch();
  auto v = std::chrono::duration_cast<std::chrono::milliseconds>(s).count();
  return v;
}

template <>
SD_INLINE SD_HOST_DEVICE float RandomGenerator::relativeT<float>(sd::LongType index) {
  u32 u;
  u._u32 = (0x3f800000 | (this->xoroshiro32(index) >> 9));
  return u._f32 - 1.0f;
}

template <>
SD_INLINE SD_HOST_DEVICE double RandomGenerator::relativeT<double>(sd::LongType index) {
#ifdef __DOUBLE_RNG__
  u64 u;
  u._ulong = ((UINT64_C(0x3FF) << 52) | (this->xoroshiro64(index) >> 12));
  return u._double - 1.0;
#else
  return (double)relativeT<float>(index);
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE uint64_t RandomGenerator::relativeT<uint64_t>(sd::LongType index) {
  return this->xoroshiro64(index);
}

template <>
SD_INLINE SD_HOST_DEVICE uint32_t RandomGenerator::relativeT<uint32_t>(sd::LongType index) {
  return this->xoroshiro32(index);
}

template <>
SD_INLINE SD_HOST_DEVICE int RandomGenerator::relativeT<int>(sd::LongType index) {
  auto r = relativeT<uint32_t>(index);
  return r <= DataTypeUtils::max<int>() ? r : r % DataTypeUtils::max<int>();
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType RandomGenerator::relativeT<sd::LongType>(sd::LongType index) {
  auto r = relativeT<uint64_t>(index);
  return r <= DataTypeUtils::max<sd::LongType>() ? r : r % DataTypeUtils::max<sd::LongType>();
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T RandomGenerator::relativeT(sd::LongType index, T from, T to) {
  auto t = this->relativeT<T>(index);
  auto z = from + T(t * (to - from));
  return z;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType RandomGenerator::relativeT(sd::LongType index, sd::LongType from,
                                                                 sd::LongType to) {
  auto t = this->relativeT<double>(index);
  auto z = from + sd::LongType(t * (to - from));
  return z;
}

template <>
SD_INLINE SD_HOST_DEVICE int RandomGenerator::relativeT(sd::LongType index, int from, int to) {
  auto t = this->relativeT<float>(index);
  auto z = from + float(t * (to - from));
  return z;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T RandomGenerator::relativeT(sd::LongType index) {
  // This is default implementation for floating point types
  return static_cast<T>(relativeT<float>(index));
}

SD_INLINE SD_HOST_DEVICE int RandomGenerator::relativeInt(sd::LongType index) {
  auto r = relativeT<uint32_t>(index);
  return r <= DataTypeUtils::max<int>() ? r : r % DataTypeUtils::max<int>();
}

SD_INLINE SD_HOST_DEVICE sd::LongType RandomGenerator::relativeLong(sd::LongType index) {
  auto r = relativeT<uint64_t>(index);
  return r <= DataTypeUtils::max<sd::LongType>() ? r : r % DataTypeUtils::max<sd::LongType>();
}

//////
static SD_INLINE SD_HOST_DEVICE uint32_t rotl(const uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

static SD_INLINE SD_HOST_DEVICE uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static SD_INLINE SD_HOST_DEVICE uint32_t next(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
  const uint32_t result = rotl(s0 + s3, 7) + s0;
  return result;
}

SD_INLINE SD_HOST_DEVICE uint32_t RandomGenerator::xoroshiro32(uint64_t index) {
  auto s0 = _rootState._ulong;
  auto s1 = _nodeState._ulong;

  // xor by idx
  s0 |= ((index + 2) * (s1 + 24243287));
  s1 ^= ((index + 2) * (s0 + 723829));

  unsigned long val = 0;
  val = s1 ^ s0;
  int *pHalf = reinterpret_cast<int *>(&val);

  return rotl(*pHalf * 0x9E3779BB, 5) * 5;
}

SD_INLINE SD_HOST_DEVICE uint64_t RandomGenerator::xoroshiro64(uint64_t index) {
  uint64_t upper = ((uint64_t)xoroshiro32(index)) << 32;
  uint32_t lower = xoroshiro32(sd::math::sd_rotl<uint64_t>(index, 32));
  return upper + lower;
}

SD_INLINE SD_HOST_DEVICE void RandomGenerator::rewindH(uint64_t steps) {
  // we only update node state, if any
  auto s0 = _nodeState._du32._v0;
  auto s1 = _nodeState._du32._v1;

  s1 ^= s0;
  _nodeState._du32._v0 = rotl(s0, 26) ^ s1 ^ (s1 << 9);  // a, b
  _nodeState._du32._v1 = rotl(s1, 13);                   // c

  _nodeState._long ^= (steps ^ 0xdeadbeef);
}
}  // namespace graph
}  // namespace sd

#endif
