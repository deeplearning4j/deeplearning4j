//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_GENERATOR_H
#define LIBND4J_HELPER_GENERATOR_H

#ifdef __GNUC__
#include <inttypes.h>
#endif


#include <mutex>

namespace nd4j {
    namespace random {

#ifdef __CUDACC__
        class CudaManaged {
        private:

        public:
            void *operator new(size_t len) {
                void *ptr;
//                cudaHostAlloc(&ptr, len, cudaHostAllocDefault);
                cudaMallocManaged(&ptr, len);
                cudaDeviceSynchronize();
                return ptr;
             }

            void operator delete(void *ptr) {
                cudaDeviceSynchronize();
                cudaFree(ptr);
//                cudaFreeHost(ptr);
            }
        };

        class RandomBuffer : public CudaManaged {
#else
        class RandomBuffer {
#endif
        private:
            long size;
            uint64_t *buffer;
            long offset;
            long seed;
            long position;
            long generation;
            long currentPosition;
            long amplifier;
            unsigned int synchronizer;

#ifdef __CUDACC__
            curandGenerator_t gen;
#endif

            std::mutex mtx;

        public:
            /**
             * This method allocates buffer of size * sizeof(long)
             *
             * @param size
             * @return
             */
#ifdef __CUDACC__
            __host__ __device__
#endif
            RandomBuffer(long seed, long size, uint64_t *buffer) {
                this->buffer = buffer;
                this->seed = seed;
                this->size = size;
                this->generation = 1;
                this->currentPosition = 0;
                this->offset = 0;
                this->amplifier = seed;
                this->synchronizer = 0;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            uint64_t *getBuffer() {
                return this->buffer;
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
            long getSize() {
                return this->size;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            long getSeed() {
                return this->seed;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void setSeed(long seed) {
                this->seed = seed;
                this->amplifier = seed;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            long getAllocatedSize() {
                return this->size * sizeof(long);
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            long getOffset() {
                return this->currentPosition;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void setOffset(long offset) {
                this->currentPosition = offset;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void reSeed(long amplifier) {
                this->amplifier = amplifier;
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            uint64_t getElement(long position) {
                long actualPosition = this->getOffset() + position;
                long tempGen = generation;
                if (actualPosition > this->size) {
                    tempGen += actualPosition / this->size;
                    actualPosition = actualPosition % this->size;
                }

                uint64_t ret = tempGen == generation ? buffer[actualPosition] : buffer[actualPosition] ^ tempGen + 11;

                if(generation > 1)
                    ret = ret ^ generation + 11;

                if (amplifier != seed)
                    ret = ret ^ amplifier + 11;

                return ret;
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
            long getNextIndex() {
                mtx.lock();
                currentPosition++;
                if (currentPosition >= size) {
                    currentPosition = 0;
                    generation++;
                }
                long ret = currentPosition;
                mtx.unlock();

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
            void rewind(long numberOfElements) {
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

					        long newPos = this->getOffset() + numberOfElements;
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
                        long newPos = this->getOffset() + numberOfElements;
                        if (newPos > this->getSize()) {
                            generation += newPos / this->size;
                            newPos = newPos % this->size;
                        } else if (newPos == this->getSize())
                            generation++;
                            newPos = 0;

                        this->setOffset(newPos);
                    }
                }
            }
        };
#else
            void rewind(long numberOfElements) {
                long newPos = this->getOffset() + numberOfElements;
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
        };

#endif

        class IGenerator {
        protected:
            long limit;
            long seed;
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
            void setOffset(long offset) {
                this->realBuffer->setOffset(offset);
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            long getElementAbsolute(long position) {
                return buffer[position];
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            long getElementRelative(long position) {
                return buffer[realBuffer->getOffset() + position];
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            virtual void refreshBuffer() = 0;
        };



        class Xoroshiro128 : public IGenerator {
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
            uint64_t seedConv(long seed) {
                uint64_t x = (uint64_t) seed;
                uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
                z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
                return z ^ (z >> 31);
            }

#ifdef __CUDACC__
            __host__ __device__
#endif
            void jump(void) {
                static const uint64_t JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };

                uint64_t s0 = 0;
                uint64_t s1 = 0;
                for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
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

                for (long i = 0; i < limit; i++) {
                    buffer[i] = next64();
                }
            }
        };
    }
}

#endif //LIBND4J_HELPER_GENERATOR_H
