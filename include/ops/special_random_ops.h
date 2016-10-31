//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIAL_RANDOM_OPS_H
#define LIBND4J_SPECIAL_RANDOM_OPS_H

#include <ops/random_ops.h>

namespace randomOps {

    template<typename T>
    class Choice {
    public:

        method_idx
        method_X
        method_XY

        static const bool requiresSpecial = true;


        static inline void specialOp(Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
            /**
             * X holds data,
             * Y holds probabilities
             * Z will hold results
             */

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
            nd4j::random::Xoroshiro128 generator(buffer);
            nd4j::random::RandomHelper<T> helper(&generator);

            // TODO: we probably might want to skip this sum, and state that probabilities array should be real probabilities, i.e. should sum to 1.0
            //T probSum = extraArguments[0];

            int xLength = shape::length(xShapeBuffer);
            int yLength = shape::length(yShapeBuffer);
            int zLength = shape::length(zShapeBuffer);

            int xEWS = shape::elementWiseStride(xShapeBuffer);
            int yEWS = shape::elementWiseStride(yShapeBuffer);
            int zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (zEWS >= 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                for (int e = 0; e < zLength; e++) {
                    T prob = helper.nextT();
                    T cumProb = (T) 0.0f;
                    for (int f; f < yLength; f++) {
                        T relProb = y[f * yEWS];
                        cumProb += relProb;

                        if (prob <= cumProb || f == yLength - 1) {
                            z[e * zEWS] = x[f * xEWS];
                        }
                    }
                }
            } else {

            }

        }
    };


    /**
    * This Op produces random values within specified boundaries. Distribuion is Gaussian
    */
    template<typename T>
    class GaussianDistribution {
    public:


        method_XY
        method_X
        method_idx

        static const bool requiresSpecial = true;


        static inline void
        specialOp(Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
            const T epsilon = (T) 1e-15;
            const T maxT = std::numeric_limits<T>::max();
            const T two_pi = (T) 2.0 * 3.14159265358979323846;

            int zLength = shape::length(zShapeBuffer);
            int zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            int span = (zLength / _threads) + 8;

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
            nd4j::random::Xoroshiro128 *generator = new nd4j::random::Xoroshiro128(buffer);
            nd4j::random::RandomHelper<T> *helper = new nd4j::random::RandomHelper<T>(generator);

            T mean = extraArguments[0];
            T stddev = extraArguments[1];

#pragma omp parallel num_threads(_threads) if (_threads > 1) proc_bind(spread)
            {
                int tid = omp_get_thread_num();
                int start = span * tid;
                int end = span * (tid + 1);
                if (end > zLength) end = zLength;

                T z0, z1;
                T u0, u1;

                bool generated = false;

                for (int e = start; e < end; e++) {
                    if (!generated) {

                        int attempt = 1;
                        do {
                            u0 = helper->relativeT(e * attempt);
                            u1 = helper->relativeT(e + (zLength * attempt));
                            attempt++;
                        } while (u0 <= epsilon );


                        z0 = nd4j::math::nd4j_sqrt<T>((T) -2.0f * nd4j::math::nd4j_log<T>(u0)) * nd4j::math::nd4j_cos<T>(two_pi * u1);
                        z1 = nd4j::math::nd4j_sqrt<T>((T) -2.0f * nd4j::math::nd4j_log<T>(u0)) * nd4j::math::nd4j_sin<T>(two_pi * u1);

                        generated = true;

                        z[e * zEWS] = z0 * stddev + mean;
                    } else {
                        z[e * zEWS] = z1 * stddev + mean;

                        generated = false;
                    }
                }
            }

            helper->rewind(zLength * 2);

            delete helper;
            delete generator;
        }
    };


    /**
    * This Op produces random values within [0..N], Distribuion is binomial
    */
    template<typename T>
    class BinomialDistribution {
    public:


        method_XY
        method_X
        method_idx

        static const bool requiresSpecial = true;


        static inline void specialOp(Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
            int trials = (int) extraArguments[0];


            int zLength = shape::length(zShapeBuffer);

            int yEWS = shape::elementWiseStride(yShapeBuffer);
            int zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            int span = (zLength / _threads) + 8;

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
            nd4j::random::Xoroshiro128 *generator = new nd4j::random::Xoroshiro128(buffer);
            nd4j::random::RandomHelper<T> *helper = new nd4j::random::RandomHelper<T>(generator);

#pragma omp parallel num_threads(_threads) if (_threads > 1) proc_bind(spread)
            {
                int tid = omp_get_thread_num();
                int start = span * tid;
                int end = span * (tid + 1);
                if (end > zLength) end = zLength;

                T prob = extraArguments[1];

                for (int e = start; e < end; e++) {

                    int success = 0;
                    for (int t = 1; t <= trials; t++) {
                        T randVal = helper->relativeT(e * t);
                        if (y != z) {
                            // we're using external probs
                            prob = y[t-1];
                        }

                        if (randVal < prob)
                            success++;
                    }

                    // if trials is set to 0, effectively we just have successful memset
                    z[e * zEWS] = (T) success;
                }
            }

            if (trials > 0)
                helper->rewind(zLength * trials);
        }
    };
}

#endif //LIBND4J_SPECIAL_RANDOM_OPS_H
