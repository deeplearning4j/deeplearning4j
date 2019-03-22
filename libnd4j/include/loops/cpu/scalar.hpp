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
// Created by raver119 on 08.10.2017.
//

#include "../scalar.h"
#include <op_boilerplate.h>
#include <types/types.h>
#include "../legacy_ops.h"

using namespace simdOps;

namespace functions {
namespace scalar    {


////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong *xShapeInfo, 
                                                void *vextraParams, 
                                                void *vz, Nd4jLong *zShapeInfo, 
                                                void *vscalars, 
                                                int *dimension, int dimensionLength, 
                                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, 
                                                Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
        
    auto x = reinterpret_cast<X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto scalars = reinterpret_cast<Y *>(vscalars);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    if (tadShapeInfoZ == nullptr) {
        tadShapeInfoZ = tadShapeInfo;
        tadOffsetsZ = tadOffsets;
    }

    // tad preparation
    int tadEws = shape::elementWiseStride(tadShapeInfo);
    int zEws = shape::elementWiseStride(zShapeInfo);
    int tadLength = shape::length(tadShapeInfo);
    int numTads =shape::length(xShapeInfo) / tadLength;

    if(tadEws < 1 || zEws < 1) {
        printf("ScalarTransform<X, Y, Z>::transform: super-bad loop visited. Shouldn't ever happen\n");
        return;
    }

    int num_threads = nd4j::math::nd4j_min<int>(numTads, omp_get_max_threads());

    if (tadEws == 1) {
        PRAGMA_OMP_PARALLEL_FOR_THREADS(num_threads)
        for (unsigned int r = 0; r < numTads; r++) {
            auto oZ = z + tadOffsetsZ[r];
            auto oX = x + tadOffsets[r];

            PRAGMA_OMP_SIMD
            for (unsigned int f = 0; f < tadLength; f++)
                oZ[f] = OpType::op(oX[f], scalars[r], extraParams);
        }
    } else {
        PRAGMA_OMP_PARALLEL_FOR_THREADS(num_threads)
        for (unsigned int r = 0; r < numTads; r++) {
            auto oZ = z + tadOffsetsZ[r];
            auto oX = x + tadOffsets[r];

            PRAGMA_OMP_SIMD
            for (unsigned int f = 0; f < tadLength; f++)
                oZ[f * zEws] = OpType::op(oX[f * tadEws], scalars[r], extraParams);
        }
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void ScalarTransform<X,Y,Z>::transform(int opNum,
                              void *x, Nd4jLong *xShapeInfo,
                              void *extraParams,
                              void *z, Nd4jLong *zShapeInfo,
                              void *scalars,
                              int *dimension, int dimensionLength,
                              Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                              Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {

    DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(const int opNum,
                                        void *x, Nd4jLong xStride,
                                        void *z, Nd4jLong zStride,
                                        void *scalar,
                                        void *extraParams,
                                        const Nd4jLong n) {

    DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xStride, z, zStride, scalar, extraParams, n), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(const int opNum,
                                        void *x, Nd4jLong *xShapeInfo,
                                        void *z, Nd4jLong *zShapeInfo,
                                        void *scalar,
                                        void *extraParams) {

    DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong *xShapeInfo,
                                        void *vz, Nd4jLong *zShapeInfo,
                                        void *vscalar,
                                        void *vextraParams) {

    auto x = reinterpret_cast<X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto scalar = reinterpret_cast<Y *>(vscalar)[0];
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    const auto len = shape::length(xShapeInfo);
    const auto xEws = shape::elementWiseStride(xShapeInfo);
    const auto zEws = shape::elementWiseStride(zShapeInfo);
    const auto xOrder = shape::order(xShapeInfo);
    const auto zOrder = shape::order(zShapeInfo);

    if(xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
        transform<OpType>(x, xEws, z, zEws, vscalar, extraParams, len);              
    }
    else {
                
        const bool xSimpe = shape::isStrideSimple(xShapeInfo);
        const bool zSimpe = shape::isStrideSimple(zShapeInfo);

        uint xShapeInfoCast[MAX_RANK];
        const bool canCastX = nd4j::DataTypeUtils::castShapeInfo<uint>(xShapeInfo, xShapeInfoCast);

        nd4j::OmpLaunchHelper info(len);

        if(shape::haveSameOffsets(xShapeInfo, zShapeInfo)) {

            PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
            {
                auto threadNum = omp_get_thread_num();                    
                auto threadOffset = info.getThreadOffset(threadNum);
                auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                PRAGMA_OMP_SIMD
                for (unsigned int i = 0; i < ulen; i++) {
                    auto offset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX);                    
                    z[offset] = OpType::op(x[offset], scalar, extraParams);
                }
            }
        }
        else {
            
            uint zShapeInfoCast[MAX_RANK];
            const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo, zShapeInfoCast);

            PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
            {
                auto threadNum = omp_get_thread_num();                    
                auto threadOffset = info.getThreadOffset(threadNum);
                auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                PRAGMA_OMP_SIMD
                for (unsigned int i = 0; i < ulen; i++) {
                    auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX);
                    auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, len, canCastZ);
                    z[zOffset] = OpType::op(x[xOffset], scalar, extraParams);
                }
            }
        }  
    }                        
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong xEws,
                                        void *vz, Nd4jLong zEws,
                                        void *vscalar,
                                        void *vextraParams,
                                        const Nd4jLong len) {
               
    auto x = reinterpret_cast<X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto scalar = reinterpret_cast<Y *>(vscalar)[0];
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    nd4j::OmpLaunchHelper info(len);

    if (xEws == 1 && zEws == 1) {

        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {
            auto threadNum = omp_get_thread_num();
            auto threadOffset = info.getThreadOffset(threadNum);
            auto xi = x + threadOffset;
            auto zi = z + threadOffset;
            auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

            PRAGMA_OMP_SIMD
            for (unsigned int i = 0; i < ulen; i++)
                zi[i] = OpType::op(xi[i], scalar, extraParams);
        }
    } else {

        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {
            auto threadNum = omp_get_thread_num();
            auto threadOffset = info.getThreadOffset(threadNum);
            auto xi = x + xEws * threadOffset;
            auto zi = z + zEws * threadOffset;
            auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

            PRAGMA_OMP_SIMD
            for (unsigned int i = 0; i < ulen; i++)
                zi[i * zEws] = OpType::op(xi[i * xEws], scalar, extraParams);
        }
    }
}



}
}
