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

#ifndef LIBND4J_RANDOM_OPS_H
#define LIBND4J_RANDOM_OPS_H

#ifdef __CUDACC__
#define random_def __device__ __host__ inline static
#else
#define random_def inline static
#endif

// since we can't inherit/overwrite static methods - we just define default impls
#define method_idx  random_def T op(Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator* rng, T *extraParams) { return -1.0f; }
#define method_X  random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator* rng, T *extraParams) { return -2.0f; }
#define method_XY  random_def T op(T valueX, T valueY, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator* rng, T *extraParams) { return -3.0f; }

#define no_exec_special static const bool requiresSpecial = false; static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) { }

#ifdef __CUDACC__
#define no_exec_special_cuda __device__ static inline void specialOpCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) { }
#else
#define no_exec_special_cuda
#endif

#include <helpers/helper_generator.h>
#include <graph/RandomGenerator.h>
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

        random_def T op(T valueX, T valueY, Nd4jLong idx,  Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
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

        random_def T op(Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
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

        random_def T op(Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            return extraParams[0] >= helper->relativeT<T>(idx) ? (T) 1.0f : (T) 0.0f;
        }

        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
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

        random_def T op(Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T lambda = extraParams[0];
            T x = helper->relativeT<T>(idx,  sd::DataTypeUtils::min<T>(), T(1.f) - sd::DataTypeUtils::template min<T>()); // x from (0, 1) without bounds
            T xVal = -sd::math::nd4j_log<T,T>(x);

            return xVal <= (T)0.f ? (T)0.f : xVal / lambda; //pow<T, T, T>((T) M_E, -(lambda * x));
        }

        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T lambda = extraParams[0];
            return valueX <= (T)0.f ? (T)0.f : (T)(valueX/lambda); //1.f - sd::math::nd4j_exp<T,T>(-lambda * valueX); //pow<T, T, T>((T) M_E, -(lambda * valueX));
        }
    };

    template <typename T>
    class PoissonDistribution {
    public:
        no_exec_special
        no_exec_special_cuda

        method_XY

        random_def T op(Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T lambda = extraParams[0];
            T x = helper->relativeT(idx, -sd::DataTypeUtils::template max<T>() / 10 , sd::DataTypeUtils::template max<T>() / 10);
            return x <= (T)0.f ? (T)0.f : sd::math::nd4j_igammac<T,T,T>(sd::math::nd4j_floor<T,T>(x), lambda);
        }

        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T lambda = extraParams[0];
            return valueX <= (T)0.f ? (T)0.f : (T)sd::math::nd4j_igammac<T,T,T>(sd::math::nd4j_floor<T,T>(valueX), lambda);
        }
    };

    template <typename T>
    class GammaDistribution {
    public:
        no_exec_special
        no_exec_special_cuda

        method_XY

        random_def T op(Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T alpha = extraParams[0];
            T beta = extraParams[1];
            T x = helper->relativeT(idx, -sd::DataTypeUtils::template max<T>() / 10 , sd::DataTypeUtils::template max<T>() / 10);
            return x <= (T)0.f ? (T)0.f : sd::math::nd4j_igamma<T,T,T>(alpha, x * beta);
        }

        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T alpha = extraParams[0];
            T beta = extraParams[1];
            return valueX <= (T)0.f ? (T)0.f : sd::math::nd4j_igamma<T,T,T>(alpha, beta * valueX);
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
        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
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
        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
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
        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
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

        random_def T op(Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T from = extraParams[0];
            T to = extraParams[1];
            T step = extraParams[2];

	        if (step == static_cast<T>(0.0f)) {
            	step = (T) idx / ((T)length - (T) 1.0f);
            	return from * ((T) 1.0f - step) + step * to;
            }
	        return from + (idx * step);

        }
    };

    template <typename T>
    class ExponentialDistributionInv {          // inverse exponential distribution
    public:
        no_exec_special
        no_exec_special_cuda

        method_XY

        random_def T op(Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T lambda = extraParams[0];
            T x = helper->relativeT(idx, sd::DataTypeUtils::template min<T>(), (T)1.f - sd::DataTypeUtils::template min<T>());
            return -sd::math::nd4j_log<T, T>((T)1.f - x) / lambda;
        }

        random_def T op(T valueX, Nd4jLong idx, Nd4jLong length, sd::graph::RandomGenerator *helper, T *extraParams) {
            T lambda = extraParams[0];            
            return -sd::math::nd4j_log<T, T>((T)1.f - valueX) / lambda;  // valueX must be within (0, 1]
        }
    };

}

#endif //LIBND4J_RANDOM_OPS_H
