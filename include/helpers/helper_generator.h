//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_GENERATOR_H
#define LIBND4J_HELPER_GENERATOR_H

#ifdef __GNUC__
#include <inttypes.h>
#endif

namespace nd4j {
    namespace random {

        class RandomBuffer {
        private:
            long size;
            long *buffer;
            long offset;
            long seed;
            long position;
            long generation;

        public:
            /**
             * This method allocates buffer of size * sizeof(long)
             *
             * @param size
             * @return
             */
            RandomBuffer(long size, long seed) {
                this->buffer = (long *) malloc(size * sizeof(long));
                this->seed = seed;
                this->generation = 1;
            }

            long *getBuffer() {
                return this->buffer;
            }

            long getSize() {
                return this->size;
            }

            long getAllocatedSize() {
                return this->size * sizeof(long);
            }

            long getOffset() {
                return this->offset;
            }

            void setOffset(long seed) {
                this->seed = seed;
            }

            long getElement(long position) {
                return buffer[position];
            }

            long getNextElement() {
                // TODO: proper implementation needed here
                return buffer[0];
            }

            ~RandomBuffer() {
                free(buffer);
            }
        };

        class IGenerator {
        protected:
            long limit = 0;
            long *buffer;
            RandomBuffer *realBuffer;

        public:

            IGenerator(RandomBuffer *buffer) {
                this->limit = buffer->getSize();
                this->buffer = buffer->getBuffer();
                this->realBuffer = buffer;
            }

            RandomBuffer *getBuffer() {
                return realBuffer;
            }

            void setOffset(long offset) {
                this->realBuffer->setOffset(offset);
            }

            long getElementAbsolute(long position) {
                return buffer[position];
            }

            long getElementRelative(long position) {
                return buffer[realBuffer->getOffset() + position];
            }

            virtual void refreshBuffer() = 0;
        };



        class Xoroshiro128 : public IGenerator {
        protected:
            uint64_t seed;
            uint64_t state[2];


            static inline uint64_t rotl(const uint64_t x, int k) {
                return (x << k) | (x >> (64 - k));
            }

            /**
             * This method returns 64 random bits
             * @return
             */
            uint64_t next64() {
                const uint64_t s0 = state[0];
                uint64_t s1 = state[1];
                const uint64_t result = s0 + s1;

                s1 ^= s0;
                state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
                state[1] = rotl(s1, 36); // c

                return result;
            }


            uint64_t seedConv(long seed) {
                uint64_t x = (uint64_t) seed;
                uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
                z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
                return z ^ (z >> 31);
            }

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
            Xoroshiro128(RandomBuffer *buffer) : IGenerator(buffer) {
                //
            }

            void refreshBuffer() {
                for (long i = 0; i < limit; i++) {
                    buffer[i] = (long) next64();
                }
            }
        };
    }
}

#endif //LIBND4J_HELPER_GENERATOR_H
