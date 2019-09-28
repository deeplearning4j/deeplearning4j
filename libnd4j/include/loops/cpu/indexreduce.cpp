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

#include "../indexreduce.h"
#include <op_boilerplate.h>
#include <Loops.h>
#include <types/types.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>
#include "../legacy_ops.h"

using namespace simdOps;

namespace functions   {
namespace indexreduce {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
Nd4jLong IndexReduce<X,Y>::execScalar( const int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams) {
    RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), INDEX_REDUCE_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void IndexReduce<X,Y>::exec(const int opNum,
                        void *x,  Nd4jLong *xShapeInfo,
                        void *extraParams,
                        void *z, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset) {

DISPATCH_BY_OPNUM_TT(exec, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffset), INDEX_REDUCE_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
template<typename OpType>
Nd4jLong IndexReduce<X, Y>::execScalar(void *vx, Nd4jLong *xShapeInfo, void *vextraParams) {

    auto x = reinterpret_cast<X *>(vx);
    auto extraParams = reinterpret_cast<X *>(vextraParams);

    //T startingVal = OpType::startingValue(x);
    auto startingIndex = OpType::startingIndexValue(x);
    auto len = shape::length(xShapeInfo);
    auto xEws = shape::elementWiseStride(xShapeInfo);
    nd4j::OmpLaunchHelper info(len);

    uint xShapeInfoCast[MAX_RANK];
    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
    const int maxThreads = nd4j::math::nd4j_min<int>(64, nd4j::Environment::getInstance()->maxThreads());
    IndexValue<X> intermediatery[64];
    for (int e = 0; e < maxThreads; e++)
        intermediatery[e].index = -1;

    if (xEws == 1) {
        auto func = PRAGMA_THREADS_FOR {
            intermediatery[thread_id] = OpType::startingIndexValue(x);

            for (auto i = start; i < stop; i += increment) {
                IndexValue<X> curr(x[i], i);
                intermediatery[thread_id] = OpType::update(intermediatery[thread_id], curr, extraParams);
            }
        };

        samediff::Threads::parallel_for(func, maxThreads, 0, len, 1);

        for (int e = 0; e < maxThreads; e++)
            if (intermediatery[e].index >= 0)
                startingIndex = OpType::update(startingIndex, intermediatery[e], extraParams);

    } else {
        auto func = PRAGMA_THREADS_FOR {
            intermediatery[thread_id] = OpType::startingIndexValue(x);

            for (auto i = start; i < stop; i += increment) {
                auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                IndexValue<X> curr(x[offset], i);
                intermediatery[thread_id] = OpType::update(intermediatery[thread_id], curr, extraParams);
            }
        };

        samediff::Threads::parallel_for(func, maxThreads, 0, len, 1);

        for (int e = 0; e < maxThreads; e++)
            if (intermediatery[e].index >= 0)
                startingIndex = OpType::update(startingIndex, intermediatery[e], extraParams);
    }
    return startingIndex.index;
}


////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void IndexReduce<X, Z>::exec(void *vx, Nd4jLong *xShapeInfo,
                        void *vextraParams,
                        void *vz, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset) {

    auto x = reinterpret_cast<X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<X *>(vextraParams);

    const Nd4jLong zLen = shape::length(zShapeInfo);

    if(nd4j::ArrayOptions::arrayType(xShapeInfo) == nd4j::ArrayType::EMPTY) {
        if(nd4j::ArrayOptions::arrayType(zShapeInfo) == nd4j::ArrayType::EMPTY)
            return;
        const auto indexValue = OpType::startingIndexValue(x);

        for (uint i = 0; i < zLen; i++)
            z[i] = (Z) indexValue.index;

        return;
    }

    if(shape::isScalar(zShapeInfo)) {
        z[0] = (Z) execScalar<OpType>(x,xShapeInfo,extraParams);
        return;
    }

    auto tadOnlyShapeInfo = tadShapeInfo;
    Nd4jLong *tadOffsets = tadOffset;

    if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
        if (dimensionLength < 1)
            return;

        auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);

        tadOnlyShapeInfo = tadPack.primaryShapeInfo();
        tadOffsets = tadPack.primaryOffsets();
    }

    nd4j::IndexReductionLoops<X,Z>::template loopIndexReduce<OpType>(x, xShapeInfo, z, zShapeInfo,  tadOnlyShapeInfo, tadOffsets, extraParams);
}


BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT IndexReduce, , LIBND4J_TYPES, INDEXING_TYPES);

}
}