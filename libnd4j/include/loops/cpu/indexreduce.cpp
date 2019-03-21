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
#include "../legacy_ops.h"

using namespace simdOps;

namespace functions   {
namespace indexreduce {

////////////////////////////////////////////////////////////////////////
template <typename X> Nd4jLong IndexReduce<X>::execScalar( const int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams) {
    RETURNING_DISPATCH_BY_OPNUM_T(execScalar, PARAMS(x, xShapeInfo, extraParams), INDEX_REDUCE_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X>
void IndexReduce<X>::exec(const int opNum,
                        void *x,  Nd4jLong *xShapeInfo,
                        void *extraParams,
                        Nd4jLong *z, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength, 
                        Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset) {

DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffset), INDEX_REDUCE_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template<typename OpType>
Nd4jLong IndexReduce<X>::execScalar(void *vx, Nd4jLong *xShapeInfo, void *vextraParams) {
        
    auto x = reinterpret_cast<X *>(vx);
    auto extraParams = reinterpret_cast<X *>(vextraParams);

    //T startingVal = OpType::startingValue(x);
    auto startingIndex = OpType::startingIndexValue(x);
    auto len = shape::length(xShapeInfo);
    auto xEws = shape::elementWiseStride(xShapeInfo);
    nd4j::OmpLaunchHelper info(len);

    uint xShapeInfoCast[MAX_RANK];
    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

    if (xEws == 1) {
        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {
            auto local = OpType::startingIndexValue(x);
            auto threadNum = omp_get_thread_num();
            auto threadOffset = info.getThreadOffset(threadNum);

            auto ulen = info.getItersPerThread(threadNum);

            for (Nd4jLong i = 0; i < ulen; i++) {
                IndexValue<X> curr(x[i + threadOffset], i + threadOffset);
                local = OpType::update(local, curr, extraParams);
            }

            PRAGMA_OMP_CRITICAL
            startingIndex = OpType::update(startingIndex, local, extraParams);
        }
    } else {
        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {
            auto local = OpType::startingIndexValue(x);
            auto threadNum = omp_get_thread_num();
            auto threadOffset = info.getThreadOffset(threadNum);

            auto ulen = info.getItersPerThread(threadNum);

            for (Nd4jLong i = 0; i < ulen; i++) {
                auto offset = shape::indexOffset(threadOffset + i, xShapeInfo, xShapeInfoCast, len, canCastX);
                IndexValue<X> curr(x[offset], threadOffset + i);
                local = OpType::update(local, curr, extraParams);
            }

            PRAGMA_OMP_CRITICAL
            startingIndex = OpType::update(startingIndex, local, extraParams);
        }
    }
    return startingIndex.index;
}


////////////////////////////////////////////////////////////////////////
template <typename X>
template<typename OpType>
void IndexReduce<X>::exec(void *vx, Nd4jLong *xShapeInfo,
                        void *vextraParams,
                        Nd4jLong *z, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset) {

    auto x = reinterpret_cast<X *>(vx);
    auto extraParams = reinterpret_cast<X *>(vextraParams);

    if(shape::isScalar(zShapeInfo)) {
        z[0] = execScalar<OpType>(x,xShapeInfo,extraParams);
        return;
    }

    const Nd4jLong zLen = shape::length(zShapeInfo);

    auto tadOnlyShapeInfo = tadShapeInfo;
    Nd4jLong *tadOffsets = tadOffset;
    shape::TAD *tad = nullptr;

    if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
        tad = new shape::TAD();
        tad->init(xShapeInfo, dimension, dimensionLength);
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        if (tad->dimensionLength < 1) {
            delete tad;
            return;
        }

        tadOnlyShapeInfo = tad->tadOnlyShapeInfo;
        tadOffsets = tad->tadOffsets;
    }

    int *dimension, int dimensionLength,
    nd4j::Loops::loopIndexTadXZ<X, OpType>(x, xShapeInfo, z, zShapeInfo, tadOnlyShapeInfo, tadOffsets, static std::vector<int> evalDimsToExclude(const int rank, const std::vector<int>& dimensions),  extraParams);
    
    if (tad != nullptr)
        delete tad;

    // int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    // int numThreads = nd4j::math::nd4j_min<int>(zLen, omp_get_max_threads());

    // uint tadOnlyShapeInfoCast[MAX_RANK];                    
    // bool canCastX = nd4j::DataTypeUtils::castShapeInfo(tadOnlyShapeInfo, tadOnlyShapeInfoCast);

    // PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
    // for(Nd4jLong i = 0; i < zLen; i++) {

    //     auto offset = tadOffsets[i];
    //     auto indexValue = OpType::startingIndexValue(&x[offset]);

    //     PRAGMA_OMP_SIMD
    //     for(int j = 0; j < tadLength; j++) {
    //         auto xOffset = offset + shape::indexOffset(j, tadOnlyShapeInfo, tadOnlyShapeInfoCast, tadLength, canCastX);
    //         IndexValue<X> comp(x[xOffset], j);
    //         indexValue = OpType::update(indexValue,comp,extraParams);
    //     }
    //     z[i] = indexValue.index;
    // }    
}


BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT IndexReduce, , LIBND4J_TYPES);

}
}