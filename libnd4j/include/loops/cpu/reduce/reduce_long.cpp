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
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <types/types.h>
#include <system/op_boilerplate.h>
#include <loops/reduce_long.h>
#include <loops/legacy_ops.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/Loops.h>
#include <helpers/ConstantTadHelper.h>

using namespace simdOps;

namespace functions {
    namespace reduce {
        template <typename X, typename Z>
        template <typename OpType>
        void _CUDA_H ReduceLongFunction<X,Z>::execScalar(const void *vx, const Nd4jLong *xShapeInfo,
                                                         void *vextraParams,
                                                         void *vz, const Nd4jLong *zShapeInfo) {
            auto x = reinterpret_cast<const X *>(vx);
            auto z = reinterpret_cast<Z *>(vz);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            const Nd4jLong length = shape::length(xShapeInfo);
            auto xEws = shape::elementWiseStride(xShapeInfo);

            if (shape::isEmpty(xShapeInfo)) {
                z[0] = OpType::startingValue(x);
                return;
            }

             if(sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
                if(sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY)
                    return;
                const auto startingVal = OpType::startingValue(x);

                for (Nd4jLong i = 0; i < length; i++)
                    z[i] = startingVal;
                return;
            }

            if (xEws >= 1) {
                z[0] = execScalar<OpType>(x, xEws, length, extraParams);
            }
            else {
                auto startingValue = OpType::startingValue(x);
                uint xShapeInfoCast[MAX_RANK];
                const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                int maxThreads = sd::math::nd4j_min<int>(64, sd::Environment::getInstance().maxThreads());
                Z intermediate[64];

                PRAGMA_OMP_SIMD
                for (auto e = 0; e < maxThreads; e++)
                    intermediate[e] = OpType::startingValue(x);

                auto func = PRAGMA_THREADS_FOR {
                    for (auto i = start; i < stop; i++)
                        intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX)], extraParams), extraParams);
                };

                maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

                // merge results
                for (int e = 1; e < maxThreads; e++)
                    intermediate[0] = OpType::update(intermediate[0], intermediate[e], extraParams);

                // write out results
                z[0] = OpType::postProcess(intermediate[0], length, extraParams);
            }
        }


        template <typename X, typename Z>
        template <typename OpType>
            Z _CUDA_H ReduceLongFunction<X, Z>::execScalar(const void *vx, const Nd4jLong *xShapeInfo, void *vextraParams) {
                auto x = reinterpret_cast<const X *>(vx);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                const Nd4jLong length = shape::length(xShapeInfo);
                auto xEws = shape::elementWiseStride(xShapeInfo);

                if (xEws >= 1) {
                    return execScalar<OpType>(x, xEws, length, extraParams);
                }
                else {
                    auto startingValue = OpType::startingValue(x);
                    uint xShapeInfoCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                    for (Nd4jLong i = 0; i < length; i++)
                        startingValue = OpType::update(startingValue, OpType::op(x[shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX)], extraParams), extraParams);

                    return OpType::postProcess(startingValue, length, extraParams);
                }
            }


        template <typename X, typename Y>
        Y ReduceLongFunction<X, Y>::execScalar(const int opNum,
                                               const void *x, const Nd4jLong *xShapeInfo,
                                               void *extraParams) {
                RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_LONG_OPS);
        }

        template <typename X, typename Y>
        void ReduceLongFunction<X, Y>::execScalar(const int opNum,
                                                  const void *x, const Nd4jLong *xShapeInfo,
                                                  void *extraParams,
                                                  void *z, const Nd4jLong *zShapeInfo) {
            DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo), REDUCE_LONG_OPS);
        }


        template <typename X, typename Z>
        template<typename OpType>
        void _CUDA_H ReduceLongFunction<X,Z>::exec(const void *x, const Nd4jLong *xShapeInfo,
                                                   void *extraParams,
                                                   void *vresult, const Nd4jLong *resultShapeInfo) {
                auto z = reinterpret_cast<Z*>(vresult);
                z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
        }

        template <typename X, typename Z>
        template <typename OpType>
        Z _CUDA_H ReduceLongFunction<X, Z>::execScalar(const void *vx, Nd4jLong xEws, Nd4jLong length, void *vextraParams) {

            auto x = reinterpret_cast<const X *>(vx);
            auto extraParams = reinterpret_cast<X *>(vextraParams);
            int maxThreads = sd::math::nd4j_min<int>(64, sd::Environment::getInstance().maxThreads());
            Z intermediate[64];

            PRAGMA_OMP_SIMD
            for (auto e = 0; e < maxThreads; e++)
                intermediate[e] = OpType::startingValue(x);

            auto func = PRAGMA_THREADS_FOR {
                if (xEws == 1) {
                    for (auto i = start; i < stop; i++)
                        intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[i], extraParams), extraParams);
                } else {
                    for (auto i = start; i < stop; i++)
                        intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[i * xEws], extraParams), extraParams);
                }
            };

            maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

            // merge results
            for (int e = 1; e < maxThreads; e++)
                intermediate[0] = OpType::update(intermediate[0], intermediate[e], extraParams);

            // return result
            return OpType::postProcess(intermediate[0], length, extraParams);
        }


////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void _CUDA_H ReduceLongFunction<X, Z>::exec(sd::memory::Workspace* workspace, const void *vx, const Nd4jLong *xShapeInfo, void *vextraParams, void *vz, const Nd4jLong *zShapeInfo, const int* dims) {

    const X* x = reinterpret_cast<const X*>(vx);
          Z* z = reinterpret_cast<Z*>(vz);
          X* extraParams = reinterpret_cast<X*>(vextraParams);

    const int xRank = shape::rank(xShapeInfo);
    const int zRank = shape::rank(zShapeInfo);

     if(sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {

        const auto startingVal = OpType::startingValue(x);
        const auto zLen = shape::length(zShapeInfo);

        for (Nd4jLong i = 0; i < zLen; i++)
            z[i] = startingVal;
        return;
    }

    if (shape::length(zShapeInfo) == 1) {
        z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
        return;
    }

    if (OpType::requiresSpecialAccumulation) {
        OpType::execSpecial(x, xShapeInfo, extraParams, z, zShapeInfo, const_cast<int*>(dims)+zRank, xRank-zRank, nullptr, nullptr);
            return;
    }

#ifdef INLINE_LOOPS
    sd::ReductionLoops<X,Z,X>::template loopReduce<OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, extraParams);
#else
    sd::ReductionLongLoops<X,Z>::template innerloopReduce<OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, extraParams);
#endif

}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void ReduceLongFunction<X, Y>::exec(const int opNum, sd::memory::Workspace* workspace, const void *vx, const Nd4jLong *xShapeInfo, void *vextraParams, void *vz, const Nd4jLong *zShapeInfo, const int *dims) {

    DISPATCH_BY_OPNUM_TT(exec, PARAMS(workspace, vx, xShapeInfo, vextraParams, vz, zShapeInfo, dims), REDUCE_LONG_OPS);
}



BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT ReduceLongFunction, , LIBND4J_TYPES, LONG_TYPES);
}
}

