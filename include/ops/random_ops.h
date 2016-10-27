//
// @author raver119@gmail.com
//

#ifndef LIBND4J_RANDOM_OPS_H
#define LIBND4J_RANDOM_OPS_H

#ifdef __CUDACC__
#define random_def __device__ inline static
#else
#define random_def inline static
#endif

// since we can't inherit/overwrite static methods - we just define default impls
#define method_idx  random_def T op(int idx, int length, nd4j::RandomHelper<T> *helper, T *extraParams) { return 0.0f; }
#define method_X  random_def T op(T valueX, int idx, nd4j::RandomHelper<T> *helper, T *extraParams) { return 0.0f; }
#define method_XY  random_def T op(T valueX, T valueY, int idx, nd4j::RandomHelper<T> *helper, T *extraParams) { return 0.0f; }

#include <helpers/helper_random.h>

namespace randomOps {

    /**
     * This Op merges two arrays per-element, if probability meets threshold
     */
    template<typename T>
    class ProbablisticMerge {
    public:

        method_idx
        method_X

        random_def T op(T valueX, T valueY, int idx, nd4j::RandomHelper<T> *helper, T *extraParams) {
            T threshold = extraParams[0];
            T randVal = helper->nextT((T) 1.0f);

            return randVal <= threshold ? valueY : valueX;
        }
    };

    /**
     * This Op produces random values within specified boundaries
     */
    template<typename T>
    class BoundedDistribution {
    public:

        method_XY
        method_X

        random_def T op(int idx, int length, nd4j::RandomHelper<T> *helper, T *extraParams) {
            return helper->nextT(extraParams[0], extraParams[1]);
        }
    };

    /**
     * Basic DropOut/DropConnect Op
     */
    template<typename T>
    class DropOut {
    public:

        method_idx
        method_XY

        random_def T op(T valueX, int idx, nd4j::RandomHelper<T> *helper, T *extraParams) {
            T randVal = helper->nextT(extraParams[0]);
            return randVal <= extraParams[1] ? (T) 0.0f : valueX;
        }
    };

    /**
     * Inverted DropOut implementation, used in DL4j
     */
    template<typename T>
    class DropOutInverted {
    public:

        method_idx
        method_XY

        random_def T op(T valueX, int idx, nd4j::RandomHelper<T> *helper, T *extraParams) {
            T prob = extraParams[1];
            T randVal = helper->nextT(extraParams[0]);
            return randVal >= prob ? (T) 0.0f : valueX / prob;
        }
    };


    template<typename T>
    class Linspace {
    public:
        method_X
        method_XY

        random_def T op(int idx, int length, nd4j::RandomHelper<T> *helper, T *extraParams) {
            T from = extraParams[0];
            T to = extraParams[1];

            T step = idx / (length - 1);

            return from * (1 - step) + step * to;
        }
    };
}

#endif //LIBND4J_RANDOM_OPS_H
