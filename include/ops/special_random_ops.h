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

            int elementsPerThread = zLength / ELEMENT_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (zEWS >= 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                for (int e = 0; e < zLength; e++) {
                    T prob = helper.nextT((T) 1.0f);
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
}

#endif //LIBND4J_SPECIAL_RANDOM_OPS_H
