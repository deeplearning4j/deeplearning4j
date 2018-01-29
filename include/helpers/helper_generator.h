//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_GENERATOR_H
#define LIBND4J_HELPER_GENERATOR_H

#include <pointercast.h>
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


#define MAX_UINT 18446744073709551615LLU


namespace nd4j {
    namespace random {

#ifdef __CUDACC__
        class ND4J_EXPORT CudaManaged {
        private:

        protected:
            void *devHolder;

        public:
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
        class ND4J_EXPORT RandomBuffer {
#endif
        private:
            void *devHolder;
            Nd4jIndex size;
            uint64_t *buffer;
            uint64_t *devBuffer;
            Nd4jIndex offset;
            Nd4jIndex seed;
            Nd4jIndex position;
            Nd4jIndex generation;
            Nd4jIndex currentPosition;
            Nd4jIndex amplifier;
            unsigned int synchronizer;

#ifdef __CUDACC__
            curandGenerator_t gen;
#endif

        public:
            /**
             * This method allocates buffer of size * sizeof(Nd4jIndex)
             *
             * @param size
             * @return
             */
#ifdef __CUDACC__
            __host__
            RandomBuffer(Nd4jIndex seed, Nd4jIndex size, uint64_t *hostBuffer, uint64_t *devBuffer) {
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
                return (Nd4jPointer) devHolder;
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
            RandomBuffer(Nd4jIndex seed, Nd4jIndex size, uint64_t *buffer) {
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

#ifdef __CUDACC__
            __host__ __device__
#endif
            inline uint64_t *getBuffer() {
                return this->buffer;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            inline uint64_t *getDeviceBuffer() {
                return this->devBuffer;
            }

#ifdef __CUDACC__
            __device__ __host__ curandGenerator_t *getGeneratorPointer() {
                return &gen;
            }

            __host__ __device__ curandGenerator_t getGenerator() {
                return gen;
            }

            __host__
            void setBuffer(uint64_t *ptr) {
                this->buffer = ptr;
            }
#endif

#ifdef __CUDACC__
            __host__ __device__
#endif
            inline Nd4jIndex getSize() {
                return this->size;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            inline Nd4jIndex getSeed() {
                return this->seed;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void setSeed(Nd4jIndex seed) {
                this->seed = seed;
                this->amplifier = seed;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            Nd4jIndex getAllocatedSize() {
                return this->size * sizeof(double);
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            inline Nd4jIndex getOffset() {
                return this->currentPosition;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void setOffset(Nd4jIndex offset) {
                this->currentPosition = offset;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void reSeed(Nd4jIndex amplifier) {
                this->amplifier = amplifier;
            }

#ifdef __CUDACC__
            __device__
#endif
            inline uint64_t getElement(Nd4jIndex position) {

                Nd4jIndex actualPosition = this->getOffset() + position;
                Nd4jIndex tempGen = generation;
                if (actualPosition >= this->size) {
                    tempGen += actualPosition / this->size;
                    actualPosition = actualPosition % this->size;
                }
#ifdef __CUDACC__
                __syncthreads();

//                int *intBuffer = (int *) devBuffer;

                uint64_t ret = (uint64_t) devBuffer[actualPosition];
#else
                uint64_t ret = (uint64_t) buffer[actualPosition];
#endif

                if (tempGen != generation)
                    ret = safeShift(ret, tempGen);

                if(generation > 1)
                    ret = safeShift(ret, generation);

                if (amplifier != seed)
                    ret = safeShift(ret, amplifier);

#ifdef __CUDACC__
                __syncthreads();
#endif
                if (amplifier != seed || generation > 1 || tempGen != generation)
                    ret = next64(seedConv((Nd4jIndex) ret));


                return ret;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            uint64_t next64(uint64_t shiftedSeed) {
                const uint64_t s0 = (uint64_t) shiftedSeed;
                uint64_t s1 = (uint64_t) shiftedSeed % MAX_INT + 11;
                uint64_t r0, r1;

                s1 ^= s0;
                r0 = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
                r1 = rotl(s1, 36); // c

                return r0 + r1;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            static inline uint64_t rotl(const uint64_t x, int k) {
                return (x << k) | (x >> (64 - k));
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            uint64_t static inline safeShift(uint64_t x, uint64_t y) {
                if (y != 0 && x > MAX_UINT / y) {
                    return x / y + 11;
                } else return (x * y) + 11;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            uint64_t seedConv(Nd4jIndex seed) {
                uint64_t x = (uint64_t) seed;
                uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
                z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
                return z ^ (z >> 31);
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void incrementGeneration() {
                this->generation++;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            Nd4jIndex getNextIndex() {
                currentPosition++;
                if (currentPosition >= size) {
                    currentPosition = 0;
                    generation++;
                }
                Nd4jIndex ret = currentPosition;

                return ret;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            uint64_t getNextElement() {
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
            void rewind(Nd4jIndex numberOfElements) {
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

					        Nd4jIndex newPos = this->getOffset() + numberOfElements;
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
                        Nd4jIndex newPos = this->getOffset() + numberOfElements;
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
            void rewindH(Nd4jIndex numberOfElements) {
                Nd4jIndex newPos = this->getOffset() + numberOfElements;
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
#ifdef __CUDACC__
            __device__
#endif
            int nextInt() {
                int r = (int) nextUInt();
                return r < 0 ? -1 * r : r;
            };

#ifdef __CUDACC__
            __device__
#endif
            uint64_t nextUInt() {
                return getNextElement();
            }

            /**
             * This method returns random int in range [0..to]
             * @param to
             * @return
             */
#ifdef __CUDACC__
            __device__
#endif
            int nextInt(int to) {
                int r = nextInt();
                int m = to - 1;
                if ((to & m) == 0)  // i.e., bound is a power of 2
                    r = (int) ((to * (Nd4jIndex) r) >> 31);
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
#ifdef __CUDACC__
            __device__
#endif
            int nextInt(int from, int to) {
                if (from == 0)
                    return nextInt(to);

                return from + nextInt(to - from);
            };


            /**
             * This method returns random T in range of [0..MAX_FLOAT]
             * @return
             */
            template<typename T>
#ifdef __CUDACC__
            __device__
#endif
            T nextMaxT() {
                T rnd = (T) getNextElement();
                return rnd < 0 ? -1 * rnd : rnd;
            }


            /**
             * This method returns random T in range of [0..1]
             * @return
             */
            template<typename T>
#ifdef __CUDACC__
            __device__
#endif
            T nextT() {
                return (T) nextUInt() / (T) MAX_UINT;
            }

            /**
             * This method returns random T in range of [0..to]
             * @param to
             * @return
             */
            template<typename T>
#ifdef __CUDACC__
            __device__
#endif
            T nextT(T to) {
                if (to == (T) 1.0f)
                    return nextT<T>();

                return nextT<T>((T) 0.0f, to);
            }

            /**
             * This method returns random T in range [from..to]
             * @param from
             * @param to
             * @return
             */
            template<typename T>
#ifdef __CUDACC__
            __device__
#endif
            T inline nextT(T from, T to) {
                return from + (nextT<T>() * (to - from));
            }


#ifdef __CUDACC__
            __device__
#endif
            inline uint64_t relativeUInt(Nd4jIndex index) {
                return getElement(index);
            }

            /**
             *  relative methods are made as workaround for lock-free concurrent execution
             */

#ifdef __CUDACC__
            __device__
#endif
            int inline relativeInt(Nd4jIndex index) {
                return (int) (relativeUInt(index) % ((unsigned int) MAX_INT + 1));
            }

            /**
             * This method returns random int within [0..to]
             *
             * @param index
             * @param to
             * @return
             */
#ifdef __CUDACC__
            __device__
#endif
            int inline relativeInt(Nd4jIndex index, int to) {
                int rel = relativeInt(index);
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
#ifdef __CUDACC__
            __device__
#endif
            int relativeInt(Nd4jIndex index, int from, int to) {
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
/*
            template <typename T>
            T relativeT(Nd4jIndex index);

            template <typename T>
            T relativeT(Nd4jIndex index, T to);

            template <typename T>
            T relativeT(Nd4jIndex index, T from, T to);

            */
            template <typename T>
#ifdef __CUDACC__
    __device__
#endif
            T relativeT(Nd4jIndex index) {
                if (sizeof(T) < 4) {
                    // FIXME: this is fast hack for short types, like fp16. This should be improved.
                    return (T)((float) relativeUInt(index) / (float) MAX_UINT);
                } else return (T) relativeUInt(index) / (T) MAX_UINT;
            }

/**
 * This method returns random T within [0..to]
 *
 * @param index
 * @param to
 * @return
 */

            template<typename T>
#ifdef __CUDACC__
            __device__
#endif
            T relativeT(Nd4jIndex index, T to) {
                if (to == (T) 1.0f)
                    return relativeT<T>(index);

                return relativeT<T>(index, (T) 0.0f, to);
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
#ifdef __CUDACC__
            __device__
#endif
            T relativeT(Nd4jIndex index, T from, T to) {
                return from + (relativeT<T>(index) * (to - from));
            }

        };

        class ND4J_EXPORT IGenerator {
        protected:
            Nd4jIndex limit;
            Nd4jIndex seed;
            uint64_t *buffer;
            nd4j::random::RandomBuffer *realBuffer;

        public:

#ifdef __CUDACC__
            __host__ __device__
#endif
            IGenerator(nd4j::random::RandomBuffer *buffer) {
                this->limit = buffer->getSize();
                this->buffer = (uint64_t *) buffer->getBuffer();
                this->realBuffer = buffer;
                this->seed = buffer->getSeed();
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            RandomBuffer *getBuffer() {
                return realBuffer;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void setOffset(Nd4jIndex offset) {
                this->realBuffer->setOffset(offset);
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            Nd4jIndex getElementAbsolute(Nd4jIndex position) {
                return buffer[position];
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            Nd4jIndex getElementRelative(Nd4jIndex position) {
                return buffer[realBuffer->getOffset() + position];
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            virtual void refreshBuffer() = 0;
        };



        class ND4J_EXPORT Xoroshiro128 : public IGenerator {
        protected:
            uint64_t state[2];

#ifdef __CUDACC__
            __host__ __device__
#endif
            static inline uint64_t rotl(const uint64_t x, int k) {
                return (x << k) | (x >> (64 - k));
            }

            /**
             * This method returns 64 random bits
             * @return
             */
#ifdef __CUDACC__
            __host__ __device__
#endif
            uint64_t next64() {
                const uint64_t s0 = state[0];
                uint64_t s1 = state[1];
                const uint64_t result = s0 + s1;

                s1 ^= s0;
                state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
                state[1] = rotl(s1, 36); // c

                return result;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            uint64_t seedConv(Nd4jIndex seed) {
                uint64_t x = (uint64_t) seed;
                uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
                z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
                return z ^ (z >> 31);
            }

#ifdef __CUDACC__
            __host__
#endif
            void jump(void) {
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
#ifdef __CUDACC__
            __host__ __device__
#endif
            Xoroshiro128(nd4j::random::RandomBuffer *buffer) : IGenerator(buffer) {
                //
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void refreshBuffer() {
                state[0] = seedConv(this->seed);
                state[1] = seedConv(this->seed * 119 + 3);

                for (Nd4jIndex i = 0; i < limit; i++) {
                    buffer[i] = next64();
                }
            }
        };
    }
}
#endif //LIBND4J_HELPER_GENERATOR_H
