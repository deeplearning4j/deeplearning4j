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
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_GENERATOR_H
#define LIBND4J_HELPER_GENERATOR_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <array/DataTypeUtils.h>
#include <dll.h>

#ifdef _MSC_VER
// include for uint64_t on MSVC
#include <stdint.h>
#elif ANDROID
#include <stdint.h>

#ifndef UINT64_C
#if defined(__LP64__)
#define UINT64_C(c)     c ## UL
#else
#define UINT64_C(c)     c ## ULL
#endif //LP64
#endif // UINT64

#endif // MSVC/ANDROID


#ifdef __GNUC__
#include <inttypes.h>
#endif

#include <helpers/IRandomGenerator.h>



#define MAX_UINT 18446744073709551615LLU


namespace nd4j {
    namespace random {

#ifdef __CUDACC__
        class ND4J_EXPORT CudaManaged : public nd4j::IRandomGenerator {
        private:

        protected:
            void *devHolder;

        public:
            virtual _CUDA_D uint32_t relativeUint32(Nd4jLong index) = 0;
            virtual _CUDA_D uint64_t relativeUint64(Nd4jLong index) = 0;
            virtual _CUDA_H void rewindH(Nd4jLong steps) = 0;

            void *operator new(size_t len) {
                void *ptr;
                cudaHostAlloc(&ptr, len, cudaHostAllocDefault);
                return ptr;
             }

            void operator delete(void *ptr) {
                cudaFreeHost(ptr);
            }
        };

        class ND4J_EXPORT RandomBuffer : public CudaManaged {
#else
        class ND4J_EXPORT RandomBuffer : public nd4j::IRandomGenerator {
#endif
        private:
            void *devHolder;
            Nd4jLong size;
            uint64_t *buffer;
            uint64_t *devBuffer;
            Nd4jLong offset;
            Nd4jLong seed;
            Nd4jLong position;
            Nd4jLong generation;
            Nd4jLong currentPosition;
            Nd4jLong amplifier;
            unsigned int synchronizer;

#ifdef __CUDACC__
            curandGenerator_t gen;
#endif

        public:
            /**
             * This method allocates buffer of size * sizeof(Nd4jLong)
             *
             * @param size
             * @return
             */
#ifdef __CUDACC__
            __host__
            RandomBuffer(Nd4jLong seed, Nd4jLong size, uint64_t *hostBuffer, uint64_t *devBuffer) {
                this->buffer = hostBuffer;
                this->seed = seed;
                this->size = size;
                this->generation = 1;
                this->currentPosition = 0;
                this->offset = 0;
                this->amplifier = seed;
                this->synchronizer = 0;
                this->devBuffer = devBuffer;

                cudaMalloc(&devHolder, sizeof(nd4j::random::RandomBuffer));
            }

            __host__
            Nd4jPointer getDevicePointer() {
                return reinterpret_cast<Nd4jPointer>(devHolder);
            }

            __host__
            ~RandomBuffer() {
                cudaFree(devHolder);
            }

            __host__
            void propagateToDevice(nd4j::random::RandomBuffer *buffer, cudaStream_t stream) {
                cudaMemcpyAsync(devHolder, buffer, sizeof(nd4j::random::RandomBuffer), cudaMemcpyHostToDevice, stream);
            }

            __host__ __device__
#endif
            RandomBuffer(Nd4jLong seed, Nd4jLong size, uint64_t *buffer) {
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

            inline _CUDA_HD uint64_t *getBuffer() {
                return this->buffer;
            }

            inline _CUDA_HD uint64_t *getDeviceBuffer() {
                return this->devBuffer;
            }

#ifdef __CUDACC__
            _CUDA_HD curandGenerator_t *getGeneratorPointer() {
                return &gen;
            }

            _CUDA_HD curandGenerator_t getGenerator() {
                return gen;
            }


            _CUDA_H void setBuffer(uint64_t *ptr) {
                this->buffer = ptr;
            }
#endif

            inline _CUDA_HD Nd4jLong getSize() {
                return this->size;
            }

            inline _CUDA_HD Nd4jLong getSeed() {
                return this->seed;
            }

            void _CUDA_HD setSeed(Nd4jLong seed) {
                this->seed = seed;
                this->amplifier = seed;
                this->generation = 1;
            }

            Nd4jLong _CUDA_HD getAllocatedSize() {
                return this->size * sizeof(double);
            }

            inline _CUDA_HD Nd4jLong getOffset() {
                return this->currentPosition;
            }

            void _CUDA_HD setOffset(Nd4jLong offset) {
                this->currentPosition = offset;
            }

            void _CUDA_HD reSeed(Nd4jLong amplifier) {
                this->amplifier = amplifier;
            }

            inline _CUDA_D uint32_t relativeUint32(Nd4jLong index) override {
                auto x = relativeUint64(index);
                return static_cast<uint32_t>(x < DataTypeUtils::max<uint32_t>() ? x : x % DataTypeUtils::max<uint32_t>());
            }

            inline _CUDA_D uint64_t relativeUint64(Nd4jLong index) override {
                return getElement(index);
            }


            inline _CUDA_D uint64_t getElement(Nd4jLong position) {
                Nd4jLong actualPosition = this->getOffset() + position;
                Nd4jLong tempGen = generation;
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

                if (tempGen != generation)
                    ret = safeShift(ret, tempGen);

                if(generation > 1)
                    ret = safeShift(ret, generation);

                if (amplifier != seed)
                    ret = safeShift(ret, amplifier);

#ifdef __CUDACC__
//                __syncthreads();
#endif
                if (amplifier != seed || generation > 1 || tempGen != generation)
                    ret = next64(seedConv(static_cast<Nd4jLong>(ret)));

                return ret;
            }

            uint64_t _CUDA_HD next64(uint64_t shiftedSeed) {
                const auto s0 = static_cast<uint64_t>(shiftedSeed);
                auto s1 = static_cast<uint64_t>(shiftedSeed) % MAX_INT + 11;
                uint64_t r0, r1;

                s1 ^= s0;
                r0 = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
                r1 = rotl(s1, 36); // c

                return r0 + r1;
            }

            static _CUDA_HD inline uint64_t rotl(const uint64_t x, uint64_t k) {
                return (x << k) | (x >> (64 - k));
            }

            uint64_t static _CUDA_HD inline safeShift(uint64_t x, uint64_t y) {
                if (y != 0 && x > MAX_UINT / y) {
                    return x / y + 11;
                } else return (x * y) + 11;
            }

            uint64_t _CUDA_HD seedConv(Nd4jLong seed) {
                uint64_t x = static_cast<uint64_t>(seed);
                uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
                z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
                return z ^ (z >> 31);
            }

            void _CUDA_HD incrementGeneration() {
                this->generation++;
            }

            Nd4jLong _CUDA_HD getNextIndex() {
                currentPosition++;
                if (currentPosition >= size) {
                    currentPosition = 0;
                    generation++;
                }
                Nd4jLong ret = currentPosition;

                return ret;
            }

            uint64_t _CUDA_HD getNextElement() {
                // TODO: proper implementation needed here
                return generation == 1 ? buffer[getNextIndex()] : buffer[getNextIndex()]  * generation;
            }


            /**
             * This method skips X elements from buffer
             *
             * @param numberOfElements number of elements to skip
             */
#ifdef __CUDACC__
            __device__
            void rewind(Nd4jLong numberOfElements) {
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

					        Nd4jLong newPos = this->getOffset() + numberOfElements;
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
                        Nd4jLong newPos = this->getOffset() + numberOfElements;
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
            void rewindH(Nd4jLong numberOfElements) {
                Nd4jLong newPos = this->getOffset() + numberOfElements;
                if (newPos > this->getSize()) {
                    generation += newPos / this->size;
                    newPos = newPos % this->size;
                }
                else if (newPos == this->getSize()) {
                    generation++;
                    newPos = 0;
                }

                this->setOffset(newPos);
            }

            /**
            * This method returns random int in range [0..MAX_INT]
            * @return
            */
            int _CUDA_D nextInt() {
                auto u = nextUInt64();
                return u <= nd4j::DataTypeUtils::max<int>() ? static_cast<int>(u) : static_cast<int>(u % nd4j::DataTypeUtils::max<int>());
            };

            uint64_t _CUDA_D nextUInt64() {
                return getNextElement();
            }

            /**
             * This method returns random int in range [0..to]
             * @param to
             * @return
             */
            int _CUDA_D nextInt(int to) {
                int r = nextInt();
                int m = to - 1;
                if ((to & m) == 0)  // i.e., bound is a power of 2
                    r = ((to * (Nd4jLong) r) >> 31);
                else {
                    for (int u = r;
                         u - (r = u % to) + m < 0;
                         u = nextInt());
                }
                return r;
            };

            /**
             * This method returns random int in range [from..to]
             * @param from
             * @param to
             * @return
             */
            int _CUDA_D nextInt(int from, int to) {
                if (from == 0)
                    return nextInt(to);

                return from + nextInt(to - from);
            };


            /**
             * This method returns random T in range of [0..1]
             * @return
             */
            template<typename T>
            _CUDA_D T nextT() {
                auto u = static_cast<float>(nextUInt64());
                auto m = static_cast<float>(nd4j::DataTypeUtils::max<uint64_t>());
                return static_cast<T>(u / m);
            }

            /**
             * This method returns random T in range of [0..to]
             * @param to
             * @return
             */
            template<typename T>
            _CUDA_D T nextT(T to) {
                if (to == static_cast<T>(1.0f))
                    return nextT<T>();

                return nextT<T>(static_cast<T>(0.0f), to);
            }

            /**
             * This method returns random T in range [from..to]
             * @param from
             * @param to
             * @return
             */
            template<typename T>
            _CUDA_D T inline nextT(T from, T to) {
                return from + (nextT<T>() * (to - from));
            }


            /**
             *  relative methods are made as workaround for lock-free concurrent execution
             */
            inline int _CUDA_D relativeInt(Nd4jLong index) {
                auto u = relativeUint64(index);
                return u <= nd4j::DataTypeUtils::max<int>() ? static_cast<int>(u) : static_cast<int>(u % nd4j::DataTypeUtils::max<int>());
            }

            /**
             * This method returns random int within [0..to]
             *
             * @param index
             * @param to
             * @return
             */
            inline int _CUDA_D relativeInt(Nd4jLong index, int to) {
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
            inline _CUDA_D int relativeInt(Nd4jLong index, int from, int to) {
                if (from == 0)
                    return relativeInt(index, to);

                return from + relativeInt(index, to - from);
            }

            /**
             * This method returns random T within [0..1]
             *
             * @param index
             * @return
             */
            template <typename T>
            inline _CUDA_D T relativeT(Nd4jLong index) {
                /**
                 * Basically we just get float u/m value, and convert into to
                 *
                 * FIXME: once we add support for additional datatypes this code must be tweaked
                 */
                auto u = static_cast<float>(relativeUint64(index));
                auto m = static_cast<float> (nd4j::DataTypeUtils::max<uint64_t>());
                return static_cast<T>(u / m);
            }

/**
 * This method returns random T within [0..to]
 *
 * @param index
 * @param to
 * @return
 */

            template<typename T>
            _CUDA_D T relativeT(Nd4jLong index, T to) {
                if (to == static_cast<T>(1.0f))
                    return relativeT<T>(index);

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
            template<typename T>
            _CUDA_D T relativeT(Nd4jLong index, T from, T to) {
                return from + (relativeT<T>(index) * (to - from));
            }

        };

        class ND4J_EXPORT IGenerator {
        protected:
            Nd4jLong limit;
            Nd4jLong seed;
            uint64_t *buffer;
            nd4j::random::RandomBuffer *realBuffer;

        public:

            _CUDA_HD IGenerator(nd4j::random::RandomBuffer *buffer) {
                this->limit = buffer->getSize();
                this->buffer = reinterpret_cast<uint64_t *>(buffer->getBuffer());
                this->realBuffer = buffer;
                this->seed = buffer->getSeed();
            }


            _CUDA_HD RandomBuffer *getBuffer() {
                return realBuffer;
            }

            _CUDA_HD void setOffset(Nd4jLong offset) {
                this->realBuffer->setOffset(offset);
            }

            _CUDA_HD Nd4jLong getElementAbsolute(Nd4jLong position) {
                return buffer[position];
            }

            _CUDA_HD Nd4jLong getElementRelative(Nd4jLong position) {
                return buffer[realBuffer->getOffset() + position];
            }

            virtual _CUDA_HD void refreshBuffer() = 0;
        };



        class ND4J_EXPORT Xoroshiro128 : public IGenerator {
        protected:
            uint64_t state[2];

            static inline _CUDA_HD uint64_t rotl(const uint64_t x, int k) {
                return (x << k) | (x >> (64 - k));
            }

            /**
             * This method returns 64 random bits
             * @return
             */
            uint64_t _CUDA_HD next64() {
                const uint64_t s0 = state[0];
                uint64_t s1 = state[1];
                const uint64_t result = s0 + s1;

                s1 ^= s0;
                state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
                state[1] = rotl(s1, 36); // c

                return result;
            }

            uint64_t _CUDA_HD seedConv(Nd4jLong seed) {
                uint64_t x = static_cast<uint64_t>(seed);
                uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
                z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
                return z ^ (z >> 31);
            }

            void _CUDA_H jump(void) {
                static const uint64_t JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };

                uint64_t s0 = 0;
                uint64_t s1 = 0;
                for(unsigned int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
                    for(int b = 0; b < 64; b++) {
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
            _CUDA_HD Xoroshiro128(nd4j::random::RandomBuffer *buffer) : IGenerator(buffer) {
                //
            }

            _CUDA_HD void refreshBuffer() {
                state[0] = seedConv(this->seed);
                state[1] = seedConv(this->seed * 119 + 3);

                int fd = 3 + 3;

                for (Nd4jLong i = 0; i < limit; i++) {
                    buffer[i] = next64();
                }
            }
        };
    }
}
#endif //LIBND4J_HELPER_GENERATOR_H
