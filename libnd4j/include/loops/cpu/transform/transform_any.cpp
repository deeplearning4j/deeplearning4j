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
//  @author  raver119@gmail.com
//

#include <op_boilerplate.h>
#include <types/types.h>
#include <loops/transform_any.h>
#include <loops/legacy_ops.h>

using namespace simdOps;

namespace functions {
    namespace transform {

        template <typename X, typename Y>
        void TransformAny<X, Y>::exec(
				int opNum,
				void *x,
				Nd4jLong *xShapeInfo,
				void *z,
				Nd4jLong *zShapeInfo,
				void *extraParams,
				Nd4jLong *tadShapeInfo,
				Nd4jLong *tadOffsets) {
                    DISPATCH_BY_OPNUM_TT(exec, PARAMS(x, xShapeInfo, z, zShapeInfo, extraParams, tadShapeInfo, tadOffsets), TRANSFORM_ANY_OPS);
		}

/////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void _CUDA_H TransformAny<X, Z>::exec(void *vx, Nd4jLong *xShapeInfo,
                                    void *vz,Nd4jLong *zShapeInfo,
                                    void *vextraParams,
                                    Nd4jLong *tadShapeInfo,Nd4jLong *tadOffsets) {

	auto x = reinterpret_cast<X *>(vx);
	auto z = reinterpret_cast<Z *>(vz);
	auto extraParams = reinterpret_cast<X *>(vextraParams);
             
    if(OpType::requiresSpecial) {
        OpType::execSpecial(x, xShapeInfo, z, zShapeInfo, extraParams, tadShapeInfo, tadOffsets);
        return;
    }

    const auto len = shape::length(xShapeInfo);
    const auto xRank  = shape::rank(xShapeInfo);
    const auto xOrder = shape::order(xShapeInfo);
    const auto zOrder = shape::order(zShapeInfo);
    const auto xEws   = shape::elementWiseStride(xShapeInfo);
    const auto zEws   = shape::elementWiseStride(zShapeInfo);
                
    const bool badCase = xEws == 0 || zEws == 0 || xOrder != zOrder;
    const bool specialCase = (xRank == shape::rank(zShapeInfo)) && shape::shapeEquals(xShapeInfo, zShapeInfo) && badCase;

    if(xRank == 1 && specialCase) {

        const auto xStride0 = shape::stride(xShapeInfo)[0];
        const auto zStride0 = shape::stride(zShapeInfo)[0];

        PRAGMA_OMP_PARALLEL_FOR
        for (int i0 = 0; i0 < len; ++i0) 
            z[i0 * zStride0] = OpType::op(x[i0 * xStride0], extraParams);
        
        return;
    }
            
    if(xRank == 2 && specialCase) {
                
        const auto xStride0 = shape::stride(xShapeInfo)[0];
        const auto xStride1 = shape::stride(xShapeInfo)[1];
        const auto zStride0 = shape::stride(zShapeInfo)[0];
        const auto zStride1 = shape::stride(zShapeInfo)[1];

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int i0 = 0; i0 < xShapeInfo[1]; ++i0) 
            for (int i1 = 0; i1 < xShapeInfo[2]; ++i1) 
                z[i0 * zStride0 + i1 * zStride1] = OpType::op(x[i0 * xStride0 + i1 * xStride1], extraParams);
        return;
    }
            
    if(xRank == 3 && specialCase) {
                
        const auto xStride0 = shape::stride(xShapeInfo)[0];
        const auto xStride1 = shape::stride(xShapeInfo)[1];
        const auto xStride2 = shape::stride(xShapeInfo)[2];
        const auto zStride0 = shape::stride(zShapeInfo)[0];
        const auto zStride1 = shape::stride(zShapeInfo)[1];
        const auto zStride2 = shape::stride(zShapeInfo)[2];

        PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
        for (int i0 = 0; i0 < xShapeInfo[1]; ++i0) 
            for (int i1 = 0; i1 < xShapeInfo[2]; ++i1)
                for (int i2 = 0; i2 < xShapeInfo[3]; ++i2)
                    z[i0 * zStride0 + i1 * zStride1 + i2 * zStride2] = OpType::op(x[i0 * xStride0 + i1 * xStride1 + i2 * xStride2], extraParams);
        return;
    }

    if(xRank == 4 && specialCase) {
                
        const auto xStride0 = shape::stride(xShapeInfo)[0];
        const auto xStride1 = shape::stride(xShapeInfo)[1];
        const auto xStride2 = shape::stride(xShapeInfo)[2];
        const auto xStride3 = shape::stride(xShapeInfo)[3];
        const auto zStride0 = shape::stride(zShapeInfo)[0];
        const auto zStride1 = shape::stride(zShapeInfo)[1];
        const auto zStride2 = shape::stride(zShapeInfo)[2];
        const auto zStride3 = shape::stride(zShapeInfo)[3];

        PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(3)
        for (int i0 = 0; i0 < xShapeInfo[1]; ++i0) 
            for (int i1 = 0; i1 < xShapeInfo[2]; ++i1)
                for (int i2 = 0; i2 < xShapeInfo[3]; ++i2)
                    for (int i3 = 0; i3 < xShapeInfo[4]; ++i3)
                        z[i0 * zStride0 + i1 * zStride1 + i2 * zStride2 + i3 * zStride3] = OpType::op(x[i0 * xStride0 + i1 * xStride1 + i2 * xStride2 + i3 * xStride3], extraParams);
        return;
    }

    if(xRank == 5 && specialCase) {
                
        const auto xStride0 = shape::stride(xShapeInfo)[0];
        const auto xStride1 = shape::stride(xShapeInfo)[1];
        const auto xStride2 = shape::stride(xShapeInfo)[2];
        const auto xStride3 = shape::stride(xShapeInfo)[3];
        const auto xStride4 = shape::stride(xShapeInfo)[4];
        const auto zStride0 = shape::stride(zShapeInfo)[0];
        const auto zStride1 = shape::stride(zShapeInfo)[1];
        const auto zStride2 = shape::stride(zShapeInfo)[2];
        const auto zStride3 = shape::stride(zShapeInfo)[3];
        const auto zStride4 = shape::stride(zShapeInfo)[4];

        PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(4)
        for (int i0 = 0; i0 < xShapeInfo[1]; ++i0) 
            for (int i1 = 0; i1 < xShapeInfo[2]; ++i1)
                for (int i2 = 0; i2 < xShapeInfo[3]; ++i2)
                    for (int i3 = 0; i3 < xShapeInfo[4]; ++i3)
                        for (int i4 = 0; i4 < xShapeInfo[5]; ++i4)
                            z[i0 * zStride0 + i1 * zStride1 + i2 * zStride2 + i3 * zStride3 + i4 * zStride4] = OpType::op(x[i0 * xStride0 + i1 * xStride1 + i2 * xStride2 + i3 * xStride3 + i4 * xStride4], extraParams);
        return;
    }

    nd4j::OmpLaunchHelper info(len);

    if (xEws == 1 && zEws == 1 && xOrder == zOrder) {

        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {
            auto threadNum = omp_get_thread_num();
            auto threadOffset = info.getThreadOffset(threadNum);

            auto tz = z + threadOffset;
            auto tx = x + threadOffset;

            PRAGMA_OMP_SIMD
            for (unsigned int i = 0; i < info.getItersPerThread(threadNum); i++)
                tz[i] = OpType::op(tx[i], extraParams);
        }
    } 
    else if (zEws == 1 && zOrder == 'c') {
        
        // this is reshape + copy edge case
        uint xShapeInfoCast[MAX_RANK];
        bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {
            auto threadNum = omp_get_thread_num();
            auto threadOffset = info.getThreadOffset(threadNum);

            auto tz = z + threadOffset;

            PRAGMA_OMP_SIMD
            for (unsigned int i = 0; i < info.getItersPerThread(threadNum); i++)
                tz[i] = OpType::op(x[shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX)], extraParams);
        }
    } 
    else if (shape::haveSameOffsets(xShapeInfo, zShapeInfo)) {
    
        uint xShapeInfoCast[MAX_RANK];
        bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {
            auto threadNum = omp_get_thread_num();
            auto threadOffset = info.getThreadOffset(threadNum);

            PRAGMA_OMP_SIMD
            for (unsigned int i = 0; i < info.getItersPerThread(threadNum); i++) {
                auto offset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX);
                z[offset] = OpType::op(x[offset], extraParams);
            }
        }
    }
    else {
        
        uint xShapeInfoCast[MAX_RANK];
        uint zShapeInfoCast[MAX_RANK];

        bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
        bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

        PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
        {
            auto threadNum = omp_get_thread_num();
            auto threadOffset = info.getThreadOffset(threadNum);

            PRAGMA_OMP_SIMD
            for (unsigned int i = 0; i < info.getItersPerThread(threadNum); i++) {
                auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX);
                auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, len, canCastZ);
                z[zOffset] = OpType::op(x[xOffset], extraParams);
            }
        }
    }
}



BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT TransformAny, , LIBND4J_TYPES, LIBND4J_TYPES);
}
}