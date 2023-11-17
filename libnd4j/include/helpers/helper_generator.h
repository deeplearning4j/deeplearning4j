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

#ifndef LIBND4J_HELPER_GENERATOR_H
#define LIBND4J_HELPER_GENERATOR_H
#include <array/DataTypeUtils.h>
#include <system/op_boilerplate.h>

#ifdef _MSC_VER
// include for uint64_t on MSVC
#include <stdint.h>
#elif ANDROID
#include <stdint.h>

#ifndef UINT64_C
#if defined(__LP64__)
#define UINT64_C(c) c##UL
#else
#define UINT64_C(c) c##ULL
#endif  // LP64
#endif  // UINT64

#endif  // MSVC/ANDROID

#ifdef __GNUC__
#include <inttypes.h>
#endif

namespace sd {
namespace random {

#ifdef __CUDACC__
class SD_LIB_EXPORT CudaManaged {
 private:
 protected:
  void *devHolder;

 public:
  void *operator new(size_t len) {
    void *ptr;
    cudaHostAlloc(&ptr, len, cudaHostAllocDefault);
    return ptr;
  }

  void operator delete(void *ptr) { cudaFreeHost(ptr); }
};

class SD_LIB_EXPORT RandomBuffer : public CudaManaged {
#else
class SD_LIB_EXPORT RandomBuffer {
#endif
 private:
  void *devHolder;
  LongType size;
  uint64_t *buffer;
  uint64_t *devBuffer;
  LongType offset;
  LongType seed;
  LongType position;
  LongType generation;
  LongType currentPosition;
  LongType amplifier;
  unsigned int synchronizer;

#ifdef __CUDACC__
  curandGenerator_t gen;
#endif

 public:
  /**
   * This method allocates buffer of size * sizeof(sd::LongType)
   *
   * @param size
   * @return
   */
#ifdef __CUDACC__
  SD_HOST
  RandomBuffer(LongType seed, LongType size, uint64_t *hostBuffer, uint64_t *devBuffer) {
    this->buffer = hostBuffer;
    this->seed = seed;
    this->size = size;
    this->generation = 1;
    this->currentPosition = 0;
    this->offset = 0;
    this->amplifier = seed;
    this->synchronizer = 0;
    this->devBuffer = devBuffer;

    cudaMalloc(&devHolder, sizeof(RandomBuffer));
  }

  SD_HOST
  Pointer getDevicePointer() { return reinterpret_cast<Pointer>(devHolder); }

  SD_HOST
  ~RandomBuffer() { cudaFree(devHolder); }

  SD_HOST
  void propagateToDevice(RandomBuffer *buffer, cudaStream_t stream) {
    cudaMemcpyAsync(devHolder, buffer, sizeof(RandomBuffer), cudaMemcpyHostToDevice, stream);
  }

  SD_HOST_DEVICE
#endif
  RandomBuffer(LongType seed, LongType size, uint64_t *buffer) {
    this->buffer = buffer;
    this->seed = seed;
    this->size = size;
    this->generation = 1;
    this->currentPosition = 0;
    this->offset = 0;
    this->amplifier = seed;
    this->synchronizer = 0;
    this->devBuffer = buffer;
  }

  SD_INLINE SD_HOST_DEVICE uint64_t *getBuffer() { return this->buffer; }

  SD_INLINE SD_HOST_DEVICE uint64_t *getDeviceBuffer() { return this->devBuffer; }

#ifdef __CUDACC__
  SD_HOST_DEVICE curandGenerator_t *getGeneratorPointer() { return &gen; }

  SD_HOST_DEVICE curandGenerator_t getGenerator() { return gen; }

  SD_HOST void setBuffer(uint64_t *ptr) { this->buffer = ptr; }
#endif

  SD_INLINE SD_HOST_DEVICE LongType getSize() { return this->size; }

  SD_INLINE SD_HOST_DEVICE LongType getSeed() { return this->seed; }

  void SD_HOST_DEVICE setSeed(LongType seed) {
    this->seed = seed;
    this->amplifier = seed;
    this->generation = 1;
  }

  LongType SD_HOST_DEVICE getAllocatedSize() { return this->size * sizeof(double); }

  SD_INLINE SD_HOST_DEVICE LongType getOffset() { return this->currentPosition; }

  void SD_HOST_DEVICE setOffset(LongType offset) { this->currentPosition = offset; }

  void SD_HOST_DEVICE reSeed(LongType amplifier) { this->amplifier = amplifier; }

  SD_INLINE SD_DEVICE uint64_t getElement(LongType position) {
    LongType actualPosition = this->getOffset() + position;
    LongType tempGen = generation;
    if (actualPosition >= this->size) {
      tempGen += actualPosition / this->size;
      actualPosition = actualPosition % this->size;
    }
#ifdef __CUDACC__
    //                __syncthreads();

    auto ret = static_cast<uint64_t>(devBuffer[actualPosition]);
#else
    auto ret = static_cast<uint64_t>(buffer[actualPosition]);
#endif

    if (tempGen != generation) ret = safeShift(ret, tempGen);

    if (generation > 1) ret = safeShift(ret, generation);

    if (amplifier != seed) ret = safeShift(ret, amplifier);

#ifdef __CUDACC__
//                __syncthreads();
#endif
    if (amplifier != seed || generation > 1 || tempGen != generation)
      ret = next64(seedConv(static_cast<LongType>(ret)));

    return ret;
  }

  uint64_t SD_HOST_DEVICE next64(uint64_t shiftedSeed) {
    const auto s0 = static_cast<uint64_t>(shiftedSeed);
    auto s1 = static_cast<uint64_t>(shiftedSeed) % DataTypeUtils::max<int>() + 11;
    uint64_t r0, r1;

    s1 ^= s0;
    r0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);  // a, b
    r1 = rotl(s1, 36);                    // c

    return r0 + r1;
  }

  static SD_HOST_DEVICE inline uint64_t rotl(const uint64_t x, uint64_t k) { return (x << k) | (x >> (64 - k)); }

  uint64_t static SD_HOST_DEVICE inline safeShift(uint64_t x, uint64_t y) {
    if (y != 0 && x > DataTypeUtils::max<uint64_t>() / y) {
      return x / y + 11;
    } else
      return (x * y) + 11;
  }

  uint64_t SD_HOST_DEVICE seedConv(LongType seed) {
    uint64_t x = static_cast<uint64_t>(seed);
    uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
  }

  void SD_HOST_DEVICE incrementGeneration() { this->generation++; }

  LongType SD_HOST_DEVICE getNextIndex() {
    currentPosition++;
    if (currentPosition >= size) {
      currentPosition = 0;
      generation++;
    }
    LongType ret = currentPosition;

    return ret;
  }

  uint64_t SD_HOST_DEVICE getNextElement() {
    // TODO: proper implementation needed here
    return generation == 1 ? buffer[getNextIndex()] : buffer[getNextIndex()] * generation;
  }

  /**
   * This method skips X elements from buffer
   *
   * @param numberOfElements number of elements to skip
   */
#ifdef __CUDACC__
  SD_DEVICE
  void rewind(LongType numberOfElements) {
    if (gridDim.x > 1) {
      __shared__ bool amLast;

      if (threadIdx.x == 0) {
        unsigned int ticket = atomicInc(&synchronizer, gridDim.x);
        amLast = (ticket == gridDim.x - 1);
      }
      __syncthreads();

      if (amLast) {
        if (threadIdx.x == 0) {
          synchronizer = 0;

          LongType newPos = this->getOffset() + numberOfElements;
          if (newPos > this->getSize()) {
            generation += newPos / this->size;
            newPos = newPos % this->size;
          } else if (newPos == this->getSize()) {
            newPos = 0;
            generation++;
          }

          this->setOffset(newPos);
        }
      }
    } else {
      if (threadIdx.x == 0) {
        LongType newPos = this->getOffset() + numberOfElements;
        if (newPos > this->getSize()) {
          generation += newPos / this->size;
          newPos = newPos % this->size;
        } else if (newPos == this->getSize()) {
          generation++;
          newPos = 0;
        }

        this->setOffset(newPos);
      }
    }
  }
#endif
  void rewindH(LongType numberOfElements) {
    LongType newPos = this->getOffset() + numberOfElements;
    if (newPos > this->getSize()) {
      generation += newPos / this->size;
      newPos = newPos % this->size;
    } else if (newPos == this->getSize()) {
      generation++;
      newPos = 0;
    }

    this->setOffset(newPos);
  }

  /**
   * This method returns random int in range [0..SD_MAX_INT]
   * @return
   */
  int SD_DEVICE nextInt() {
    auto u = nextUInt64();
    return u <= DataTypeUtils::max<int>() ? static_cast<int>(u)
                                              : static_cast<int>(u % DataTypeUtils::max<int>());
  };

  uint64_t SD_DEVICE nextUInt64() { return getNextElement(); }

  /**
   * This method returns random int in range [0..to]
   * @param to
   * @return
   */
  int SD_DEVICE nextInt(int to) {
    int r = nextInt();
    int m = to - 1;
    if ((to & m) == 0)  // i.e., bound is a power of 2
      r = ((to * (LongType)r) >> 31);
    else {
      for (int u = r; u - (r = u % to) + m < 0; u = nextInt())
        ;
    }
    return r;
  };

  /**
   * This method returns random int in range [from..to]
   * @param from
   * @param to
   * @return
   */
  int SD_DEVICE nextInt(int from, int to) {
    if (from == 0) return nextInt(to);

    return from + nextInt(to - from);
  };

  /**
   * This method returns random T in range of [0..1]
   * @return
   */
  template <typename T>
  SD_DEVICE T nextT() {
    auto u = static_cast<float>(nextUInt64());
    auto m = static_cast<float>(DataTypeUtils::max<uint64_t>());
    return static_cast<T>(u / m);
  }

  /**
   * This method returns random T in range of [0..to]
   * @param to
   * @return
   */
  template <typename T>
  SD_DEVICE T nextT(T to) {
    if (to == static_cast<T>(1.0f)) return nextT<T>();

    return nextT<T>(static_cast<T>(0.0f), to);
  }

  /**
   * This method returns random T in range [from..to]
   * @param from
   * @param to
   * @return
   */
  template <typename T>
  SD_DEVICE T inline nextT(T from, T to) {
    return from + (nextT<T>() * (to - from));
  }

  SD_INLINE SD_DEVICE uint64_t relativeUInt64(LongType index) { return getElement(index); }

  /**
   *  relative methods are made as workaround for lock-free concurrent execution
   */
  inline int SD_DEVICE relativeInt(LongType index) {
    auto u = relativeUInt64(index);
    return u <= DataTypeUtils::max<int>() ? static_cast<int>(u)
                                              : static_cast<int>(u % DataTypeUtils::max<int>());
  }

  /**
   * This method returns random int within [0..to]
   *
   * @param index
   * @param to
   * @return
   */
  inline int SD_DEVICE relativeInt(LongType index, int to) {
    auto rel = relativeInt(index);
    return rel % to;
  }

  /**
   * This method returns random int within [from..to]
   *
   * @param index
   * @param to
   * @param from
   * @return
   */
  SD_INLINE SD_DEVICE int relativeInt(LongType index, int from, int to) {
    if (from == 0) return relativeInt(index, to);

    return from + relativeInt(index, to - from);
  }

  /**
   * This method returns random T within [0..1]
   *
   * @param index
   * @return
   */
  template <typename T>
  SD_INLINE SD_DEVICE T relativeT(LongType index) {
    /**
     * Basically we just get float u/m value, and convert into to
     *
     * FIXME: once we add support for additional datatypes this code must be tweaked
     */
    auto u = static_cast<float>(relativeUInt64(index));
    auto m = static_cast<float>(DataTypeUtils::max<uint64_t>());
    return static_cast<T>(u / m);
  }

  /**
   * This method returns random T within [0..to]
   *
   * @param index
   * @param to
   * @return
   */

  template <typename T>
  SD_DEVICE T relativeT(LongType index, T to) {
    if (to == static_cast<T>(1.0f)) return relativeT<T>(index);

    return relativeT<T>(index, static_cast<T>(0.0f), to);
  }

  /**
   * This method returns random T within [from..to]
   *
   * @param index
   * @param from
   * @param to
   * @return
   */
  template <typename T>
  SD_DEVICE T relativeT(LongType index, T from, T to) {
    return from + (relativeT<T>(index) * (to - from));
  }
};

class SD_LIB_EXPORT IGenerator {
 protected:
  LongType limit;
  LongType seed;
  uint64_t *buffer;
  RandomBuffer *realBuffer;

 public:
  SD_HOST_DEVICE IGenerator(RandomBuffer *buffer) {
    this->limit = buffer->getSize();
    this->buffer = reinterpret_cast<uint64_t *>(buffer->getBuffer());
    this->realBuffer = buffer;
    this->seed = buffer->getSeed();
  }

  SD_HOST_DEVICE RandomBuffer *getBuffer() { return realBuffer; }

  SD_HOST_DEVICE void setOffset(LongType offset) { this->realBuffer->setOffset(offset); }

  SD_HOST_DEVICE LongType getElementAbsolute(LongType position) { return buffer[position]; }

  SD_HOST_DEVICE LongType getElementRelative(LongType position) {
    return buffer[realBuffer->getOffset() + position];
  }

  virtual SD_HOST_DEVICE void refreshBuffer() = 0;
};

class SD_LIB_EXPORT Xoroshiro128 : public IGenerator {
 protected:
  uint64_t state[2];

  static SD_INLINE SD_HOST_DEVICE uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

  /**
   * This method returns 64 random bits
   * @return
   */
  uint64_t SD_HOST_DEVICE next64() {
    const uint64_t s0 = state[0];
    uint64_t s1 = state[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);  // a, b
    state[1] = rotl(s1, 36);                    // c

    return result;
  }

  uint64_t SD_HOST_DEVICE seedConv(LongType seed) {
    uint64_t x = static_cast<uint64_t>(seed);
    uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
  }

  void SD_HOST jump(void) {
    static const uint64_t JUMP[] = {0xbeac0467eba5facb, 0xd86b048b86aa9922};

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for (unsigned int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
      for (int b = 0; b < 64; b++) {
        if (JUMP[i] & 1ULL << b) {
          s0 ^= state[0];
          s1 ^= state[1];
        }
        next64();
      }

    state[0] = s0;
    state[1] = s1;
  }

 public:
  SD_HOST_DEVICE Xoroshiro128(RandomBuffer *buffer) : IGenerator(buffer) {
    //
  }

  SD_HOST_DEVICE void refreshBuffer() {
    state[0] = seedConv(this->seed);
    state[1] = seedConv(this->seed * 119 + 3);

    int fd = 3 + 3;

    for (LongType i = 0; i < limit; i++) {
      buffer[i] = next64();
    }
  }
};
}  // namespace random
}  // namespace sd
#endif  // LIBND4J_HELPER_GENERATOR_H
