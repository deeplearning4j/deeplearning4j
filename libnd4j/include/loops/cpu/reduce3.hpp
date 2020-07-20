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
#include <system/op_boilerplate.h>
#include <loops/reduce3.h>
#include <loops/legacy_ops.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/Loops.h>
#include <execution/Threads.h>

using namespace simdOps;

namespace functions {
namespace reduce3   {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::execScalar(const void *vx, const Nd4jLong *xShapeInfo,
                              void *vextraParams,
                              const void *vy, const Nd4jLong *yShapeInfo,
                              void *vz, const Nd4jLong *zShapeInfo) {

    auto x = reinterpret_cast<const X *>(vx);
    auto y = reinterpret_cast<const X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    auto length = shape::length(xShapeInfo);
    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto yEws = shape::elementWiseStride(yShapeInfo);

    if(sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY || sd::ArrayOptions::arrayType(yShapeInfo) == sd::ArrayType::EMPTY) {
        if(sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY)
            return;
        const auto startingVal = OpType::startingValue(x);

        for (Nd4jLong i = 0; i < length; i++)
            z[i] = startingVal;

        return;
    }

    Z extraParamsVals[3] = {(Z) 0.0f, (Z) 0.0f, (Z) 0.0f};

    uint xShapeInfoCast[MAX_RANK];
    const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

    Z startingVal = OpType::startingValue(x);
    int maxThreads = sd::math::nd4j_min<int>(64, sd::Environment::getInstance().maxThreads());
    Z intermediate[64];
    Z extraParamsLocal[3 * 64];

    PRAGMA_OMP_SIMD
    for (int e = 0; e < maxThreads; e++)
        intermediate[e] = startingVal;

    memset(extraParamsLocal, 0, 3 * 64 * sizeof(Z));
    if (extraParams != nullptr) {
        PRAGMA_OMP_SIMD
        // mostly for future reference
        for (int e = 0; e < maxThreads; e++) {
            extraParamsLocal[3 * e] = extraParams[0];
            extraParamsLocal[3 * e + 1] = extraParams[1];
            extraParamsLocal[3 * e + 2] = extraParams[2];
        }
    }

    sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXZ(xShapeInfo, yShapeInfo);

    if (kindOfLoop == sd::LoopKind::EWS1) {
        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[i], y[i], extraParamsLocal + 3 * thread_id), extraParamsLocal + 3 * thread_id);
            }
        };

        maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

    } else if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[offset], y[offset], extraParamsLocal + 3 * thread_id), extraParamsLocal + 3 * thread_id);
            }
        };

        maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);
    } else {
        uint yShapeInfoCast[MAX_RANK];
        const bool canCastY = sd::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
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

    // writing out result
    z[0] = OpType::postProcess(startingVal, length, extraParamsVals);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::execScalar(const int opNum,
                              const void *vx, const Nd4jLong *xShapeInfo,
                              void *extraParamsVals,
                              const void *vy, const Nd4jLong *yShapeInfo,
                              void *vz, const Nd4jLong *zShapeInfo) {

    DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::exec(const void *vx, const Nd4jLong *xShapeInfo,
                        void *vextraParams,
                        const void *vy, const Nd4jLong *yShapeInfo,
                        void *vz, const Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        int64_t start, int64_t stop) {

    auto x = reinterpret_cast<const X*>(vx);
    auto y = reinterpret_cast<const X*>(vy);
    auto z = reinterpret_cast<Z*>(vz);
    auto extraParams = reinterpret_cast<Z*>(vextraParams);

    if(shape::isScalar(zShapeInfo)) {
        execScalar<OpType>(vx, xShapeInfo, vextraParams, vy, yShapeInfo, vz, zShapeInfo);
        return;
    }
#ifdef INLINE_LOOPS
    sd::Reduction3Loops<X,Z>::template loopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, extraParams, start, stop);
#else
    sd::Reduction3Loops<X,Z>::template innerloopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, extraParams, start, stop);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::exec(const void *vx, const Nd4jLong *xShapeInfo,
                        void *vextraParams,
                        const void *vy, const Nd4jLong *yShapeInfo,
                        void *vz, const Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                        int64_t start, int64_t stop) {

    auto x = reinterpret_cast<const X *>(vx);
    auto y = reinterpret_cast<const X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);
#ifdef INLINE_LOOPS
    sd::Reduction3Loops<X,Z>::template loopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, extraParams, start, stop);
#else
    sd::Reduction3Loops<X,Z>::template innerloopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, extraParams, start, stop);
#endif
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>:: execAll(const void *vx, const Nd4jLong *xShapeInfo,
                            void *vextraParams,
                            const void *vy, const Nd4jLong *yShapeInfo,
                            void *vz, const Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                            const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                            int64_t start, int64_t stop) {

    auto x = reinterpret_cast<const X *>(vx);
    auto y = reinterpret_cast<const X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z*>(vextraParams);

#ifdef INLINE_LOOPS
    sd::Reduction3Loops<X,Z>::template loopReduce3All<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets, extraParams, start, stop);
#else
    sd::Reduction3Loops<X,Z>::template innerloopReduce3All<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets, extraParams, start, stop);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::exec(const int opNum,
                        const void *vx, const Nd4jLong *xShapeInfo,
                        void *extraParamsVals,
                        const void *vy, const Nd4jLong *yShapeInfo,
                        void *vz, const Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        int64_t start, int64_t stop) {

    DISPATCH_BY_OPNUM_TT(exec, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension, dimensionLength, start, stop), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::exec(const int opNum,
                        const void *vx, const Nd4jLong *xShapeInfo,
                        void *extraParamsVals,
                        const void *vy, const Nd4jLong *yShapeInfo,
                        void *vz, const Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                        int64_t start, int64_t stop) {

    DISPATCH_BY_OPNUM_TT(exec, PARAMS(vx,xShapeInfo,extraParamsVals,vy, yShapeInfo,vz,zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, start, stop), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::execAll(const int opNum,
                           const void *vx, const Nd4jLong *xShapeInfo,
                           void *extraParamsVals,
                           const void *vy, const Nd4jLong *yShapeInfo,
                           void *vz, const Nd4jLong *zShapeInfo,
                           int *dimension, int dimensionLength,
                           const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                           const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                           int64_t start, int64_t stop) {

    DISPATCH_BY_OPNUM_TT(execAll, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets, start, stop), REDUCE3_OPS);
}

}
}