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
        long seed;
        long state;

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
            return 0.0f;
        }
    };
}

#endif //LIBND4J_HELPER_RANDOM_H
