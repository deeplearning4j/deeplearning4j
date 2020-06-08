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
// Created by raver on 4/9/2018.
//

#include <loops/indexreduce.h>
#include <system/op_boilerplate.h>
#include <helpers/Loops.h>
#include <types/types.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>
#include <loops/legacy_ops.h>

using namespace simdOps;

namespace functions   {
namespace indexreduce {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
Nd4jLong IndexReduce<X,Y>::execScalar( const int opNum, const void *x, const Nd4jLong *xShapeInfo, void *extraParams) {
    RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), INDEX_REDUCE_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void IndexReduce<X,Y>::exec(const int opNum,
                            const void *x, const Nd4jLong *xShapeInfo,
                            void *extraParams,
                            void *z, const Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffset) {
    DISPATCH_BY_OPNUM_TT(exec, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffset), INDEX_REDUCE_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
template<typename OpType>
Nd4jLong IndexReduce<X, Y>::execScalar(const void *vx, const Nd4jLong *xShapeInfo, void *vextraParams) {

    auto x = reinterpret_cast<const X *>(vx);
    auto extraParams = reinterpret_cast<X *>(vextraParams);

    //T startingVal = OpType::startingValue(x);
    auto startingIndex = OpType::startingIndexValue(x);
    auto len = shape::length(xShapeInfo);
    auto xEws = shape::elementWiseStride(xShapeInfo);
    sd::OmpLaunchHelper info(len);

    uint xShapeInfoCast[MAX_RANK];
    bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
    int maxThreads = sd::math::nd4j_min<int>(64, sd::Environment::getInstance().maxThreads());
    IndexValue<X> intermediatery[64];
    for (int e = 0; e < maxThreads; e++)
        intermediatery[e].index = -1;

    if (xEws == 1 && shape::order(xShapeInfo) == 'c') {
        auto func = PRAGMA_THREADS_FOR {
            intermediatery[thread_id] = OpType::startingIndexValue(x);

            for (auto i = start; i < stop; i++) {
                IndexValue<X> curr(x[i], i);
                intermediatery[thread_id] = OpType::update(intermediatery[thread_id], curr, extraParams);
            }
        };

        maxThreads = samediff::Threads::parallel_for(func, 0, len, 1, maxThreads);

        for (int e = 0; e < maxThreads; e++)
            startingIndex = OpType::update(startingIndex, intermediatery[e], extraParams);

    } else {
        auto func = PRAGMA_THREADS_FOR {
            intermediatery[thread_id] = OpType::startingIndexValue(x);

            for (auto i = start; i < stop; i++) {
                auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                IndexValue<X> curr(x[offset], i);
                intermediatery[thread_id] = OpType::update(intermediatery[thread_id], curr, extraParams);
            }
        };

        maxThreads = samediff::Threads::parallel_for(func, 0, len, 1, maxThreads);

        for (int e = 0; e < maxThreads; e++)
            startingIndex = OpType::update(startingIndex, intermediatery[e], extraParams);
    }
    return startingIndex.index;
}


////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void IndexReduce<X, Z>::exec(const void *vx, const Nd4jLong *xShapeInfo,
                             void *vextraParams,
                             void *vz, const Nd4jLong *zShapeInfo,
                             int *dimension, int dimensionLength,
                             const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffset) {

    auto x = reinterpret_cast<const X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<X *>(vextraParams);

    const Nd4jLong zLen = shape::length(zShapeInfo);

    if(sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
        if(sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY)
            return;
        const auto indexValue = OpType::startingIndexValue(x);

        for (Nd4jLong i = 0; i < zLen; i++)
            z[i] = (Z) indexValue.index;

        return;
    }

    if(shape::isScalar(zShapeInfo)) {
        z[0] = (Z) execScalar<OpType>(x,xShapeInfo,extraParams);
        return;
    }

    auto tadOnlyShapeInfo = tadShapeInfo;
    auto tadOffsets = tadOffset;

    if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
        if (dimensionLength < 1)
            return;

        auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);

        tadOnlyShapeInfo = tadPack.primaryShapeInfo();
        tadOffsets = tadPack.primaryOffsets();
    }

    sd::IndexReductionLoops<X,Z>::template loopIndexReduce<OpType>(x, xShapeInfo, z, zShapeInfo,  tadOnlyShapeInfo, tadOffsets, extraParams);
}

}
}