/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <types/u64.h>
#include <types/u32.h>
#include <system/pointercast.h>
#include <system/op_boilerplate.h>
#include <system/dll.h>
#include <chrono>
#include <array/DataTypeUtils.h>
#include <helpers/logger.h>
#include <stdexcept>
#include <math/templatemath.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace sd {
    namespace graph {
#ifdef __CUDACC__
        class ND4J_EXPORT CudaManagedRandomGenerator {
        private:

        protected:
            void *devHolder;

        public:
            void *operator new(size_t len) {
                void *ptr;
                auto res = cudaHostAlloc(&ptr, len, cudaHostAllocDefault);
                if (res != 0)
                    throw std::runtime_error("CudaManagedRandomGenerator: failed to allocate memory");

                return ptr;
             }

            void operator delete(void *ptr) {
                cudaFreeHost(ptr);
            }
        };

        class ND4J_EXPORT RandomGenerator : public CudaManagedRandomGenerator {
#else
        class ND4J_EXPORT RandomGenerator {
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
            static FORCEINLINE Nd4jLong currentMilliseconds();

        public:
            FORCEINLINE _CUDA_HD uint32_t xoroshiro32(uint64_t index);
            FORCEINLINE _CUDA_HD uint64_t xoroshiro64(uint64_t index);

            /**
             * This method returns integer value between 0 and MAX_UINT
             */
            //uint32_t relativeUInt32(Nd4jLong index);

        public:
            FORCEINLINE RandomGenerator(Nd4jLong rootSeed = 0, Nd4jLong nodeSeed = 0);

            /**
             * This method allows to change graph-level state in runtime.
             * PLEASE NOTE: this method will change state of node as well.
             */
            FORCEINLINE _CUDA_H void setStates(Nd4jLong rootSeed, Nd4jLong nodeState = 0);

            

            /**
             * This method returns T value between from and to
             */
            template <typename T>
            FORCEINLINE _CUDA_HD T relativeT(Nd4jLong index, T from, T to);

            /**
             * This method returns T value between 0 and MAX_T
             */
            template <typename T>
            FORCEINLINE _CUDA_HD T relativeT(Nd4jLong index);

            /**
             * These two methods are made for JVM
             * @param index
             * @return
             */
            FORCEINLINE _CUDA_HD int relativeInt(Nd4jLong index);
            FORCEINLINE _CUDA_HD Nd4jLong relativeLong(Nd4jLong index);

            FORCEINLINE _CUDA_HD void rewindH(uint64_t steps);

            /**
             * These methods set up only node states, with non-changed root ones
             */
            FORCEINLINE _CUDA_H void setSeed(int seed) {
                _nodeState._ulong = static_cast<uint64_t>(seed);
            }

            FORCEINLINE _CUDA_H void setSeed(uint64_t seed) {
                _nodeState._ulong = seed;
            }

            FORCEINLINE _CUDA_HD Nd4jLong rootState() {
                return _rootState._long;
            }

            FORCEINLINE _CUDA_HD Nd4jLong nodeState() {
                return _nodeState._long;
            }
        };


        FORCEINLINE RandomGenerator::RandomGenerator(Nd4jLong rootSeed, Nd4jLong nodeSeed) {
            // this seed is used graph-level state
            if (rootSeed == 0)
                rootSeed = currentMilliseconds();

            // graph-level state is just first seed
            _rootState._long = rootSeed;

            // used to build second, node state
            _nodeState._long = (nodeSeed != 0 ? nodeSeed: 1298567341LL);
        }

        FORCEINLINE void RandomGenerator::setStates(Nd4jLong rootSeed, Nd4jLong nodeSeed) {
            // this seed is used graph-level state
            if (rootSeed == 0)
                rootSeed = currentMilliseconds();

            // graph-level state is just first seed
            _rootState._long = rootSeed;

            // used to build second, node state
            _nodeState._long = (nodeSeed != 0 ? nodeSeed: 1298567341LL);
        }

        FORCEINLINE Nd4jLong RandomGenerator::currentMilliseconds() {
            auto s = std::chrono::system_clock::now().time_since_epoch();
            auto v = std::chrono::duration_cast<std::chrono::milliseconds>(s).count();
            return v;
        }

        template <>
        _CUDA_HD FORCEINLINE float RandomGenerator::relativeT<float>(Nd4jLong index) {
            u32 u;
            u._u32 = (0x3f800000 | (this->xoroshiro32(index) >> 9));
            return u._f32 - 1.0f;
        }

        template <>
        _CUDA_HD FORCEINLINE double RandomGenerator::relativeT<double>(Nd4jLong index) {
#ifdef __DOUBLE_RNG__
          u64 u;
          u._ulong = ((UINT64_C(0x3FF) << 52) | (this->xoroshiro64(index) >> 12));
          return u._double - 1.0;
#else
          return (double) relativeT<float>(index);
#endif
        }

        template <>
        _CUDA_HD FORCEINLINE uint64_t RandomGenerator::relativeT<uint64_t>(Nd4jLong index) {
            return this->xoroshiro64(index);
        }

        template <>
        _CUDA_HD FORCEINLINE uint32_t RandomGenerator::relativeT<uint32_t>(Nd4jLong index) {            
            return this->xoroshiro32(index);
        }

        template <>
        _CUDA_HD FORCEINLINE int RandomGenerator::relativeT<int>(Nd4jLong index) {
            auto r = relativeT<uint32_t>(index);
            return r <= DataTypeUtils::max<int>() ? r : r % DataTypeUtils::max<int>();
        }

        template <>
        _CUDA_HD FORCEINLINE Nd4jLong RandomGenerator::relativeT<Nd4jLong>(Nd4jLong index) {
            auto r = relativeT<uint64_t>(index);
            return r <= DataTypeUtils::max<Nd4jLong>() ? r : r % DataTypeUtils::max<Nd4jLong>();
        }

        template <typename T>
        _CUDA_HD FORCEINLINE T RandomGenerator::relativeT(Nd4jLong index, T from, T to) {            
            auto t = this->relativeT<T>(index);
            auto z = from + T(t * (to - from));
            return z;
        }

        template <>
        _CUDA_HD FORCEINLINE Nd4jLong RandomGenerator::relativeT(Nd4jLong index, Nd4jLong from, Nd4jLong to) {
            auto t = this->relativeT<double>(index);
            auto z = from + Nd4jLong(t * (to - from));
            return z;
        }

        template <>
        _CUDA_HD FORCEINLINE int RandomGenerator::relativeT(Nd4jLong index, int from, int to) {
            auto t = this->relativeT<float>(index);
            auto z = from + float(t * (to - from));
            return z;
        }

        template <typename T>
        _CUDA_HD FORCEINLINE T RandomGenerator::relativeT(Nd4jLong index) {
            // This is default implementation for floating point types
            return static_cast<T>(relativeT<float>(index));
        }


        _CUDA_HD FORCEINLINE int RandomGenerator::relativeInt(Nd4jLong index) {
            auto r = relativeT<uint32_t>(index);
            return r <= DataTypeUtils::max<int>() ? r : r % DataTypeUtils::max<int>();
        }

        _CUDA_HD FORCEINLINE Nd4jLong RandomGenerator::relativeLong(Nd4jLong index) {
            auto r = relativeT<uint64_t>(index);
            return r <= DataTypeUtils::max<Nd4jLong>() ? r : r % DataTypeUtils::max<Nd4jLong>();
        }

        //////
        static FORCEINLINE _CUDA_HD uint32_t rotl(const uint32_t x, int k) {
            return (x << k) | (x >> (32 - k));
        }

        static FORCEINLINE _CUDA_HD  uint64_t rotl(const uint64_t x, int k) {
            return (x << k) | (x >> (64 - k));
        }

        static FORCEINLINE _CUDA_HD uint32_t next(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
          const uint32_t result = rotl(s0 + s3, 7) + s0;
          return result;
        }

        _CUDA_HD FORCEINLINE uint32_t RandomGenerator::xoroshiro32(uint64_t index) {
            auto s0 = _rootState._ulong;
            auto s1 = _nodeState._ulong;

            // xor by idx
            s0 |= ((index + 2) * (s1 + 24243287));
            s1 ^= ((index + 2) * (s0 + 723829));

            unsigned long val = 0;
            val = s1 ^ s0;
            int* pHalf = reinterpret_cast<int*>(&val);

            return rotl(*pHalf * 0x9E3779BB, 5) * 5;
        }

        _CUDA_HD FORCEINLINE uint64_t RandomGenerator::xoroshiro64(uint64_t index) {
            uint64_t upper = ((uint64_t) xoroshiro32(index)) << 32;
            uint32_t lower = xoroshiro32(sd::math::nd4j_rotl<uint64_t>(index, 32));
            return upper + lower;
        }

        _CUDA_HD FORCEINLINE void RandomGenerator::rewindH(uint64_t steps) {
          // we only update node state, if any
          auto s0 = _nodeState._du32._v0;
          auto s1 = _nodeState._du32._v1;

          s1 ^= s0;
          _nodeState._du32._v0 = rotl(s0, 26) ^ s1 ^ (s1 << 9); // a, b
          _nodeState._du32._v1 = rotl(s1, 13); // c

          _nodeState._long ^= (steps ^ 0xdeadbeef);
        }
    }
}

#endif
