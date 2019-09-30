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
#include <loops/reduce_same.h>
#include <loops/legacy_ops.h>
#include <OmpLaunchHelper.h>
#include <chrono>
#include <helpers/Loops.h>
#include <helpers/ConstantTadHelper.h>

using namespace simdOps;

namespace functions {
    namespace reduce {
        template <typename X>
        template <typename OpType>
        void _CUDA_H ReduceSameFunction<X>::execScalar(void *vx,
                                                Nd4jLong *xShapeInfo,
                                                void *vextraParams,
                                                void *vz,
                                                Nd4jLong *zShapeInfo) {
            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<X *>(vz);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            const auto length = shape::length(xShapeInfo);
            const auto xEws = shape::elementWiseStride(xShapeInfo);
            const int rank = shape::rank(xShapeInfo);

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
                const int maxThreads = nd4j::math::nd4j_min<int>(64, nd4j::Environment::getInstance()->maxThreads());
                X intermediate[64];

                for (int e = 0; e < maxThreads; e++)
                    intermediate[e] = start;

                uint xShapeInfoCast[MAX_RANK];
                const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                auto func = PRAGMA_THREADS_FOR {
                    for (auto i = start; i < stop; i += increment)
                        intermediate[omp_get_thread_num()] = OpType::update(intermediate[omp_get_thread_num()], OpType::op(x[shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX)], extraParams), extraParams);
                };

                samediff::Threads::parallel_for(func, maxThreads, 0, length);

                for (int e = 0; e < maxThreads; e++)
                    start = OpType::update(start, intermediate[e], extraParams);

                z[0] = OpType::postProcess(start, length, extraParams);
            }
        }


        template <typename X>
        template <typename OpType>
            X _CUDA_H ReduceSameFunction<X>::execScalar(void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vextraParams) {
                auto x = reinterpret_cast<X *>(vx);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                const Nd4jLong length = shape::length(xShapeInfo);
                const auto xEws = shape::elementWiseStride(xShapeInfo);

                if (xEws >= 1) {
                    return execScalar<OpType>(x, xEws, length, extraParams);
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
                            intermediate[omp_get_thread_num()] = OpType::update(intermediate[omp_get_thread_num()], OpType::op(x[shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX)], extraParams), extraParams);
                    };

                    samediff::Threads::parallel_for(func, maxThreads, 0, length);

                    for (int e = 0; e < maxThreads; e++)
                        start = OpType::update(start, intermediate[e], extraParams);

                    return OpType::postProcess(start, shape::length(xShapeInfo), extraParams);
                }
            }

        template <typename X>
        X ReduceSameFunction<X>::execScalar(const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams) {
                RETURNING_DISPATCH_BY_OPNUM_T(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_SAME_OPS);
        }

        template <typename X>
        void ReduceSameFunction<X>::execScalar(const int opNum,
                                        void *x,
                                        Nd4jLong *xShapeInfo,
                                        void *extraParams,
                                        void *z,
                                        Nd4jLong *zShapeInfo) {
            DISPATCH_BY_OPNUM_T(execScalar, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo), REDUCE_SAME_OPS);
        }

        template <typename X>
        void ReduceSameFunction<X>::exec(const int opNum,
                             void *x,
                             Nd4jLong *xShapeInfo,
                             void *extraParams,
                             void *z,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset) {
                DISPATCH_BY_OPNUM_T(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParams,
                                               z,
                                               zShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset),
                                  REDUCE_SAME_OPS);
        }

        template <typename X>
        template <typename OpType>
        void _CUDA_H ReduceSameFunction<X>::exec(void *vx,
                             Nd4jLong *xShapeInfo,
                             void *vextraParams,
                             void *vz,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset) {

                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<X *>(vz);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                auto zLength = shape::length(zShapeInfo);

                if(nd4j::ArrayOptions::arrayType(xShapeInfo) == nd4j::ArrayType::EMPTY) {
                    if(nd4j::ArrayOptions::arrayType(zShapeInfo) == nd4j::ArrayType::EMPTY)
                        return;
                    const auto startingVal = OpType::startingValue(x);

                    for (uint i = 0; i < zLength; i++)
                        z[i] = startingVal;
                    return;
                }

                //pre squeezed: this is for keeping the pointer to the original
                //shape information for tad offset
                //the squeezed information doesn't render the right strides for
                //tad offset
                // || tad.wholeThing
                if (zLength == 1 || dimension == nullptr || dimensionLength == shape::rank(xShapeInfo)) {
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
                nd4j::ReductionLoops<X,X,X>::template loopReduce<OpType>(x, xShapeInfo, z, zShapeInfo,  tadOnlyShapeInfo, tadOffsets, extraParams);
#else
                nd4j::ReductionSameLoops<X>::template innerloopReduce<OpType>(x, xShapeInfo, z, zShapeInfo, tadOnlyShapeInfo, tadOffsets, extraParams);
#endif
            }


        template <typename X>
        template<typename OpType>
        void _CUDA_H ReduceSameFunction<X>::exec(void *x,
                             Nd4jLong *xShapeInfo,
                             void *extraParams,
                             void *vz,
                             Nd4jLong *zShapeInfo) {
                // FIXME: wtf???
                auto z = reinterpret_cast<X*>(vz);
                z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
        }

        template <typename X>
        template <typename OpType>
        X _CUDA_H ReduceSameFunction<X>::execScalar(void *vx, Nd4jLong xEws, Nd4jLong length, void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            auto startingVal = OpType::startingValue(x);
            auto maxThreads = nd4j::math::nd4j_min(nd4j::Environment::getInstance()->maxThreads(), 64);
            X intermediatery[64];
            for (int e = 0; e < maxThreads; e++)
                intermediatery[e] = startingVal;

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


        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT ReduceSameFunction, , LIBND4J_TYPES);
    }
}