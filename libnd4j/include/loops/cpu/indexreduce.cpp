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
    Nd4jLong len = shape::length(xShapeInfo);
    int xEws = shape::elementWiseStride(xShapeInfo);

    if(xEws < 1) {        
        for (Nd4jLong i = 0; i < len; i++) {
            IndexValue<X> curr(i, x[shape::getIndexOffset(i, xShapeInfo, len)]);
            startingIndex = OpType::update(startingIndex, curr, extraParams);
        }
        return startingIndex.index;
    }
    
    if (len < ELEMENT_THRESHOLD) {
// FIXME: proper reduction to be used here
        for (Nd4jLong i = 0; i < len; i++) {
            IndexValue<X> curr(i, x[i*xEws]);
            startingIndex = OpType::update(startingIndex, curr, extraParams);
        }
        return startingIndex.index;
    }
    
    // xEws >= 1 && len >= ELEMENT_THRESHOLD
    BlockInformation info(len, ELEMENT_THRESHOLD);
#pragma omp parallel num_threads(info.threads) if (info.threads > 1) default(shared)
    {
        auto local = OpType::startingIndexValue(x);
        auto i = omp_get_thread_num();            
        Nd4jLong itemsToLoop = (i < info.threads-1) ? info.items : info.items + info.remainder;
        Nd4jLong index = i * info.items;
        auto xi = x + xEws * index;

        for (Nd4jLong j = 0; j < itemsToLoop; j++) {
            IndexValue<X> curr(index + j, xi[j*xEws]);
            local = OpType::update(local, curr, extraParams);
        }

#pragma omp critical
        startingIndex = OpType::update(startingIndex, local, extraParams);
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
    auto startingIndex = new IndexValue<X>[zLen];

#pragma omp parallel for schedule(guided) if (zLen > TAD_THRESHOLD) default(shared)
    for (Nd4jLong i = 0; i < zLen; i++)
        startingIndex[i] = OpType::startingIndexValue(x);        

    auto tadOnlyShapeInfo = tadShapeInfo;
    Nd4jLong *tadOffsets = tadOffset;
    shape::TAD *tad = nullptr;

    if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
        tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        if (tad->dimensionLength < 1) {
            delete tad;
            delete[] startingIndex;
            return;
        }

        tadOnlyShapeInfo = tad->tadOnlyShapeInfo;
        tadOffsets = tad->tadOffsets;
    }

    int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    int numTads = shape::length(xShapeInfo) / tadLength;

    if(!(shape::elementWiseStride(tadOnlyShapeInfo) > 0 && (numTads == 1 || shape::isVector(tadOnlyShapeInfo) || shape::isScalar(tadOnlyShapeInfo)))) {
        /**
         * The element wise stride belong longs to a reduction index.
         * When used out of order, we can get rid of the data
         * dependencies and rely on using the max dimension
         * specified for stride instead.
         * Say we take the sum(0,1) along long arr
         * we can use arr.stride(1) as a representation
         * along long which to iterate.
         */

        auto tadShapeShapeInfo = tadOnlyShapeInfo;
        auto xShape = shape::shapeOf(tadShapeShapeInfo);
        auto xStride = shape::stride(tadShapeShapeInfo);
        int rank = shape::rank(tadShapeShapeInfo);

#pragma omp  parallel for schedule(guided) if (zLen > TAD_THRESHOLD) default(shared)
        for(Nd4jLong i = 0; i < zLen; i++) {

            auto offset = tadOffsets[i];
            auto indexValue = OpType::startingIndexValue(&x[offset]);

            for(int j = 0; j < tadLength; j++) {
                auto xOffset = offset + shape::getIndexOffset(j, tadShapeShapeInfo, tadLength);
                IndexValue<X> comp(j, x[xOffset]);
                indexValue = OpType::update(indexValue,comp,extraParams);
            }
            z[i] = indexValue.index;
        }
    } 
    else {
        auto tadEws = shape::elementWiseStride(tadOnlyShapeInfo);

//#pragma omp parallel for schedule(guided) if (zLen > TAD_THRESHOLD) default(shared)
        for(Nd4jLong i = 0;  i < zLen; i++) {
            auto baseOffset = tadOffsets[i];
            auto indexValue = OpType::startingIndexValue(&x[baseOffset]);
// FIXME: proper reduction required here
            for(int j = 0; j < tadLength; j++) {
                IndexValue<X> comp(j, x[baseOffset + tadEws * j]);
                indexValue = OpType::update(indexValue,comp,extraParams);
            }
            z[i] = indexValue.index;
        }
    }

    delete[] startingIndex;
}


BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT IndexReduce, , LIBND4J_TYPES);

}
}