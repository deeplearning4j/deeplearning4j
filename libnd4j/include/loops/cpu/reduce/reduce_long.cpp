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
#include <ShapeUtils.h>
#include <op_boilerplate.h>
#include <loops/reduce_long.h>
#include <loops/legacy_ops.h>
#include <OmpLaunchHelper.h>
#include <helpers/Loops.h>
#include <helpers/ConstantTadHelper.h>

using namespace simdOps;

namespace functions {
    namespace reduce {
        template <typename X, typename Z>
        template <typename OpType>
        void _CUDA_H ReduceLongFunction<X,Z>::execScalar(void *vx,
                                                Nd4jLong *xShapeInfo,
                                                void *vextraParams,
                                                void *vz,
                                                Nd4jLong *zShapeInfo) {
            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<Z *>(vz);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            const Nd4jLong length = shape::length(xShapeInfo);
            auto xEws = shape::elementWiseStride(xShapeInfo);

            if (shape::isEmpty(xShapeInfo)) {
                z[0] = OpType::startingValue(x);
                return;
            }

             if(nd4j::ArrayOptions::arrayType(xShapeInfo) == nd4j::ArrayType::EMPTY) {
                if(nd4j::ArrayOptions::arrayType(zShapeInfo) == nd4j::ArrayType::EMPTY)
                    return;
                const auto startingVal = OpType::startingValue(x);

                for (uint i = 0; i < length; i++)
                    z[i] = startingVal;
                return;
            }

            if (xEws >= 1) {
                z[0] = execScalar<OpType>(x, xEws, length, extraParams);
            }
            else {
                X start = OpType::startingValue(x);
                const int maxThreads = nd4j::math::nd4j_min<int>(64, omp_get_max_threads());
                X intermediate[64];

                for (int e = 0; e < maxThreads; e++)
                    intermediate[e] = start;

                uint xShapeInfoCast[MAX_RANK];
                const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                auto func = PRAGMA_THREADS_FOR {
                    for (auto i = start; i < stop; i += increment)
                        intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX)], extraParams), extraParams);
                };

                samediff::Threads::parallel_for(func, maxThreads, 0, length);

                for (int e = 0; e < maxThreads; e++)
                    start = OpType::update(start, intermediate[e], extraParams);

                z[0] = OpType::postProcess(start, shape::length(xShapeInfo), extraParams);
            }
        }


        template <typename X, typename Z>
        template <typename OpType>
            Z _CUDA_H ReduceLongFunction<X, Z>::execScalar(void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vextraParams) {
                auto x = reinterpret_cast<X *>(vx);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                const Nd4jLong length = shape::length(xShapeInfo);
                auto xEws = shape::elementWiseStride(xShapeInfo);

                if (xEws >= 1) {
                    return execScalar<OpType>(x, xEws, length, extraParams);
                }
                else {
                    X start = OpType::startingValue(x);
                    auto maxThreads = nd4j::math::nd4j_max<int>(1, nd4j::Environment::getInstance()->maxThreads());
                    auto intermediate = new X[maxThreads];
                    for (int e = 0; e < maxThreads; e++)
                        intermediate[e] = start;

                    uint xShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                    auto func = PRAGMA_THREADS_FOR {
                        for (auto i = start; i < stop; i += increment)
                            intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX)], extraParams), extraParams);
                    };

                    samediff::Threads::parallel_for(func, maxThreads, 0, length);

                    for (int e = 0; e < maxThreads; e++)
                        start = OpType::update(start, intermediate[e], extraParams);

                    delete[] intermediate;
                    return OpType::postProcess(start, shape::length(xShapeInfo), extraParams);
                }
            }


        template <typename X, typename Y>
        Y ReduceLongFunction<X, Y>::execScalar(const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams) {
                RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_LONG_OPS);
        }

        template <typename X, typename Y>
        void ReduceLongFunction<X, Y>::execScalar(const int opNum,
                                        void *x,
                                        Nd4jLong *xShapeInfo,
                                        void *extraParams,
                                        void *z,
                                        Nd4jLong *zShapeInfo) {
            DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo), REDUCE_LONG_OPS);
        }

        template <typename X, typename Y>
        void ReduceLongFunction<X, Y>::exec(const int opNum,
                             void *x,
                             Nd4jLong *xShapeInfo,
                             void *extraParams,
                             void *z,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset) {
                DISPATCH_BY_OPNUM_TT(exec, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffset), REDUCE_LONG_OPS);
        }

        template <typename X, typename Z>
        template <typename OpType>
        void _CUDA_H ReduceLongFunction<X,Z>::exec(void *vx,
                             Nd4jLong *xShapeInfo,
                             void *vextraParams,
                             void *vresult,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset) {

                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<Z *>(vresult);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                auto resultLength = shape::length(zShapeInfo);

                if(nd4j::ArrayOptions::arrayType(xShapeInfo) == nd4j::ArrayType::EMPTY) {
                    if(nd4j::ArrayOptions::arrayType(zShapeInfo) == nd4j::ArrayType::EMPTY)
                        return;
                    const auto startingVal = OpType::startingValue(x);

                    for (uint i = 0; i < resultLength; i++)
                        z[i] = startingVal;
                    return;
                }

                //pre squeezed: this is for keeping the pointer to the original
                //shape information for tad offset
                //the squeezed information doesn't render the right strides for
                //tad offset
                // || tad.wholeThing
                if (resultLength == 1 || dimension == nullptr || dimensionLength == shape::rank(xShapeInfo)) {
                    z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
                    return;
                }

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecial(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffset);
                    return;
                }

                auto tadOnlyShapeInfo = tadShapeInfo;
                auto tadOffsets = tadOffset;

                if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
                    if (dimensionLength < 1)
                        return;

                    auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);
                    tadOnlyShapeInfo = tadPack.primaryShapeInfo();
                    tadOffsets = tadPack.primaryOffsets();
                }

#ifdef INLINE_LOOPS
                nd4j::ReductionLoops<X,Z,X>::template loopReduce<OpType>(x, xShapeInfo, z, zShapeInfo,  tadOnlyShapeInfo, tadOffsets, extraParams);
#else
                nd4j::ReductionLongLoops<X,Z>::template innerloopReduce<OpType>(x, xShapeInfo, z, zShapeInfo,  tadOnlyShapeInfo, tadOffsets, extraParams);
#endif
            }


        template <typename X, typename Z>
        template<typename OpType>
        void _CUDA_H ReduceLongFunction<X,Z>::exec(void *x,
                             Nd4jLong *xShapeInfo,
                             void *extraParams,
                             void *vresult,
                             Nd4jLong *resultShapeInfo) {
                // FIXME: wtf???
                auto z = reinterpret_cast<Z*>(vresult);
                z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
        }

        template <typename X, typename Z>
        template <typename OpType>
        Z _CUDA_H ReduceLongFunction<X, Z>::execScalar(void *vx, Nd4jLong xEws, Nd4jLong length, void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            auto startingVal = OpType::startingValue(x);

            auto maxThreads = nd4j::math::nd4j_min(nd4j::Environment::getInstance()->maxThreads(), 64);
            Z intermediatery[64];

            if (xEws == 1) {
                auto func = PRAGMA_THREADS_FOR {
                    for (auto i = start; i < stop; i += increment) {
                        intermediatery[thread_id] = OpType::update(intermediatery[thread_id], OpType::op(x[i], extraParams), extraParams);
                    }
                };

                auto actual_threads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);
                for (int e = 0; e < actual_threads; e++)
                    startingVal = OpType::update(startingVal, intermediatery[e], extraParams);

            } else {
                auto func = PRAGMA_THREADS_FOR {
                    for (auto i = start; i < stop; i += increment)
                        intermediatery[thread_id] = OpType::update(intermediatery[thread_id], OpType::op(x[i * xEws], extraParams), extraParams);
                };

                auto actual_threads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);
                for (int e = 0; e < actual_threads; e++)
                    startingVal = OpType::update(startingVal, intermediatery[e], extraParams);
            }

            return OpType::postProcess(startingVal, length, extraParams);
        }


        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT ReduceLongFunction, , LIBND4J_TYPES, LONG_TYPES);
    }
}