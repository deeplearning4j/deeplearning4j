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

// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.11.2018


#include <types/types.h>
#include <op_boilerplate.h>
#include <loops/reduce3.h>
#include <loops/legacy_ops.h>
#include <helpers/ConstantTadHelper.h>
#include <Loops.h>
#include <execution/Threads.h>

using namespace simdOps;

namespace functions {
namespace reduce3   {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::execScalar(void *vx, Nd4jLong *xShapeInfo,
                                    void *vextraParams,
                                    void *vy, Nd4jLong *yShapeInfo,
                                    void *vz, Nd4jLong *zShapeInfo) {

    auto x = reinterpret_cast<X *>(vx);
    auto y = reinterpret_cast<X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    auto length = shape::length(xShapeInfo);
    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto yEws = shape::elementWiseStride(yShapeInfo);

    if(nd4j::ArrayOptions::arrayType(xShapeInfo) == nd4j::ArrayType::EMPTY || nd4j::ArrayOptions::arrayType(yShapeInfo) == nd4j::ArrayType::EMPTY) {
        if(nd4j::ArrayOptions::arrayType(zShapeInfo) == nd4j::ArrayType::EMPTY)
            return;
        const auto startingVal = OpType::startingValue(x);

        for (uint i = 0; i < length; i++)
            z[i] = startingVal;

        return;
    }

    Z extraParamsVals[3] = {(Z) 0.0f, (Z) 0.0f, (Z) 0.0f};
    // it's possible case for EqualsWithEps op
    if (extraParams != nullptr)
        extraParamsVals[2] = extraParams[0];

    uint xShapeInfoCast[MAX_RANK];
    const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

    Z startingVal = OpType::startingValue(x);
    int maxThreads = nd4j::math::nd4j_min<int>(64, nd4j::Environment::getInstance()->maxThreads());
    nd4j::OmpLaunchHelper t(length, maxThreads);
    Z intermediate[64];
    Z extraParamsLocal[3 * 64];

    PRAGMA_OMP_SIMD
    for (int e = 0; e < maxThreads; e++)
        intermediate[e] = startingVal;

    memset(extraParamsLocal, 0, 3 * 64 * sizeof(Z));
    if (extraParams != nullptr) {
        PRAGMA_OMP_SIMD
        for (int e = 0; e < maxThreads; e++)
            extraParamsLocal[3 * e + 2] = extraParams[0];
    }

    nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXZ(xShapeInfo, yShapeInfo);

    if (kindOfLoop == nd4j::LoopKind::EWS1) {
        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment) {
                intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[i], y[i], extraParamsLocal + 3 * thread_id), extraParamsLocal + 3 * thread_id);
            }
        };

        maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

    } else if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment) {
                auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[offset], y[offset], extraParamsLocal + 3 * thread_id), extraParamsLocal + 3 * thread_id);
            }
        };

        maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);
    } else {
        uint yShapeInfoCast[MAX_RANK];
        const bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment) {
                auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                auto yOffset = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, canCastY);
                intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[xOffset], y[yOffset], extraParamsLocal + 3 * thread_id), extraParamsLocal + 3 * thread_id);
            }
        };

        maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);
    }

    // merge step
    for (int e = 0; e < maxThreads; e++)
        OpType::aggregateExtraParams(extraParamsVals, extraParamsLocal + 3 * e);
    for (int e = 0; e < maxThreads; e++)
        startingVal = OpType::update(startingVal, intermediate[e], extraParamsVals);

    z[0] = OpType::postProcess(startingVal, length, extraParamsVals);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::execScalar(const int opNum,
                                        void *vx, Nd4jLong *xShapeInfo,
                                        void *extraParamsVals,
                                        void *vy, Nd4jLong *yShapeInfo,
                                        void *vz, Nd4jLong *zShapeInfo) {

    DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::exec(void *vx, Nd4jLong *xShapeInfo,
                    void *vextraParams,
                    void *vy, Nd4jLong *yShapeInfo,
                    void *vz, Nd4jLong *zShapeInfo,
                    int *dimension, int dimensionLength) {

    auto x = reinterpret_cast<X*>(vx);
    auto y = reinterpret_cast<X*>(vy);
    auto z = reinterpret_cast<Z*>(vz);
    auto extraParams = reinterpret_cast<Z*>(vextraParams);

    if(shape::isScalar(zShapeInfo)) {
        execScalar<OpType>(vx, xShapeInfo, vextraParams, vy, yShapeInfo, vz, zShapeInfo);
        return;
    }
#ifdef INLINE_LOOPS
    nd4j::Reduction3Loops<X,Z>::template loopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, extraParams);
#else
    nd4j::Reduction3Loops<X,Z>::template innerloopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, extraParams);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::exec(void *vx, Nd4jLong *xShapeInfo,
                        void *vextraParams,
                        void *vy, Nd4jLong *yShapeInfo,
                        void *vz, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    auto x = reinterpret_cast<X *>(vx);
    auto y = reinterpret_cast<X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);
#ifdef INLINE_LOOPS
    nd4j::Reduction3Loops<X,Z>::template loopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, extraParams);
#else
    nd4j::Reduction3Loops<X,Z>::template innerloopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, extraParams);
#endif
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>:: execAll(void *vx, Nd4jLong *xShapeInfo,
                            void *vextraParams,
                            void *vy, Nd4jLong *yShapeInfo,
                            void *vz, Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets,
                            Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

    auto x = reinterpret_cast<X *>(vx);
    auto y = reinterpret_cast<X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z*>(vextraParams);

#ifdef INLINE_LOOPS
    nd4j::Reduction3Loops<X,Z>::template loopReduce3All<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets, extraParams);
#else
    nd4j::Reduction3Loops<X,Z>::template innerloopReduce3All<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets, extraParams);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::exec( const int opNum,
                        void *vx, Nd4jLong *xShapeInfo,
                        void *extraParamsVals,
                        void *vy, Nd4jLong *yShapeInfo,
                        void *vz, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength) {

    DISPATCH_BY_OPNUM_TT(exec, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension, dimensionLength), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::exec( const int opNum,
                        void *vx, Nd4jLong *xShapeInfo,
                        void *extraParamsVals,
                        void *vy, Nd4jLong *yShapeInfo,
                        void *vz, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    DISPATCH_BY_OPNUM_TT(exec, PARAMS(vx,xShapeInfo,extraParamsVals,vy, yShapeInfo,vz,zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::execAll(const int opNum,
                            void *vx, Nd4jLong *xShapeInfo,
                            void *extraParamsVals,
                            void *vy, Nd4jLong *yShapeInfo,
                            void *vz, Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets,
                            Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

    DISPATCH_BY_OPNUM_TT(execAll, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets), REDUCE3_OPS);
}




BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT Reduce3, , LIBND4J_TYPES, FLOAT_TYPES);

}
}