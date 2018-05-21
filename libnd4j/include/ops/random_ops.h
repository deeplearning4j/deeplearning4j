//
// @author raver119@gmail.com
//

#ifndef LIBND4J_RANDOM_OPS_H
#define LIBND4J_RANDOM_OPS_H

#ifdef __CUDACC__
#define random_def __device__ __host__ inline static
#else
#define random_def inline static
#endif

// since we can't inherit/overwrite static methods - we just define default impls
#define method_idx  random_def T op(Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) { return -1.0f; }
#define method_X  random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) { return -2.0f; }
#define method_XY  random_def T op(T valueX, T valueY, Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) { return -3.0f; }

#define no_exec_special static const bool requiresSpecial = false; static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) { }

#ifdef __CUDACC__
#define no_exec_special_cuda __device__ static inline void specialOpCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) { }
#else
#define no_exec_special_cuda
#endif

#include <helpers/helper_generator.h>
#include <array/DataTypeUtils.h>

namespace randomOps {

    /**
     * This Op merges two arrays per-element, if probability meets threshold
     */
    template<typename T>
    class ProbablisticMerge {
    public:

        no_exec_special
        no_exec_special_cuda

        method_idx
        method_X

        random_def T op(T valueX, T valueY, Nd4jLong idx,  Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            T threshold = extraParams[0];
            T randVal = helper->relativeT<T>(idx);

            return randVal <= threshold ? valueY : valueX;
        }
    };

    /**
     * This Op produces random values within specified boundaries. Disribution is uniform
     */
    template<typename T>
    class UniformDistribution {
    public:

        no_exec_special
        no_exec_special_cuda

        method_XY
        method_X

        random_def T op(Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            return helper->relativeT<T>(idx, extraParams[0], extraParams[1]);
        }
    };

    /**
     * This op produces single bernoulli trial
     */
    template <typename T>
    class BernoulliDistribution {
    public:
        no_exec_special
        no_exec_special_cuda

        method_XY

        random_def T op(Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            return extraParams[0] >= helper->relativeT<T>(idx) ? (T) 1.0f : (T) 0.0f;
        }

        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            return valueX >= helper->relativeT<T>(idx) ? (T) 1.0f : (T) 0.0f;
        }
    };


    /**
     * This op produces single bernoulli trial
     */
    template <typename T>
    class ExponentialDistribution {
    public:
        no_exec_special
        no_exec_special_cuda

        method_XY

        random_def T op(Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            T lambda = extraParams[0];
            T x = helper->relativeT(idx, nd4j::DataTypeUtils::template min<T>(), (T) 1.0f);
            return (T) 1.f - nd4j::math::nd4j_pow<T>((T) M_E, -(lambda * x));
        }

        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            T lambda = extraParams[0];
            return valueX <= (T) 0.f ? (T) 0.f : (T) 1.f - nd4j::math::nd4j_pow<T>((T) M_E, -(lambda * valueX));
        }
    };


    /**
     * Basic DropOut/DropConnect Op
     */
    template<typename T>
    class DropOut {
    public:

        no_exec_special
        no_exec_special_cuda

        method_idx
        method_XY

        // please note: prob is chance to retain original value
        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            T randVal = helper->relativeT<T>(idx);
            return randVal >= extraParams[0] ? (T) 0.0f : valueX;
        }
    };

    template<typename T>
    class AlphaDropOut {
    public:

        no_exec_special
        no_exec_special_cuda

        method_idx
        method_XY

        // please note: prob is chance to retain original value
        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            T randVal = helper->relativeT<T>(idx);
            // extraParams[0] == p
            // [1] = a
            // [2] = b
            // [3] = alphaPrime
            return randVal >= extraParams[0] ? (T) extraParams[1] * extraParams[3] + extraParams[2] : extraParams[1] * valueX  + extraParams[2];
        }
    };

    /**
     * Inverted DropOut implementation, used in DL4j
     */
    template<typename T>
    class DropOutInverted {
    public:

        no_exec_special
        no_exec_special_cuda

        method_idx
        method_XY

        // please note: prob is chance to retain original value
        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            T prob = extraParams[0];
            T randVal = helper->relativeT<T>(idx);
            return randVal >= prob ? (T) 0.0f : valueX / prob;
        }
    };


    template<typename T>
    class Linspace {
    public:

        no_exec_special
        no_exec_special_cuda

        method_X
        method_XY

        random_def T op(Nd4jLong idx, Nd4jLong length, nd4j::random::RandomBuffer *helper, T *extraParams) {
            T from = extraParams[0];
            T to = extraParams[1];

            T step = (T) idx / ((T)length - (T) 1.0f);

            return from * ((T) 1.0f - step) + step * to;
        }
    };
}

#endif //LIBND4J_RANDOM_OPS_H
