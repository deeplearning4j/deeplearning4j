//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_RANDOM_H
#define LIBND4J_HELPER_RANDOM_H

#ifdef __CUDACC__
#include <curand.h>
#endif

namespace nd4j {

    template <typename T>
    class RandomHelper {
    private:
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
        RandomHelper(long seed) {
            this->seed = seed;
        }

#ifdef __CUDACC__
        // for cuda we're using cuRand
        RandomHelper(long seed, long curand) {

        }
#endif

        /**
         * This method returns random int in range [0..MAX_INT]
         * @return
         */
        int nextInt() {
            return 0;
        };

        /**
         * This method returns random int in range [0..to]
         * @param to
         * @return
         */
        int nextInt(int to) {
            return 0;
        };

        /**
         * This method returns random int in range [from..to]
         * @param from
         * @param to
         * @return
         */
        int nextInt(int from, int to) {
            if (from == 0)
                return nextInt(to);

            return 0;
        };

        /**
         * This method returns random T in range of [0..MAX_FLOAT]
         * @return
         */
        T nextT() {
            return 0.0f;
        };

        /**
         * This method returns random T in range of [0..to]
         * @param to
         * @return
         */
        T nextT(T to) {
            return 0.0f;
        };

        /**
         * This method returns random T in range [from..to]
         * @param from
         * @param to
         * @return
         */
        T nextT(T from, T to) {
            if (from == (T) 0.0f)
                return nextInt(to);

            return 0.0f;
        }

    };
}

#endif //LIBND4J_HELPER_RANDOM_H
