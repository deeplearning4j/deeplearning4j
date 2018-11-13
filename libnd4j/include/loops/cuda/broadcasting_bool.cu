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
//

#include <op_boilerplate.h>
#include <loops/broadcasting_bool.h>
#include <loops/legacy_ops.h>
#include <types/types.h>
#include <Environment.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <StringUtils.h>

using namespace simdOps;

template<typename X, typename Z, typename OpClass>
static __device__ void broadcastBoolSimpleGeneric(
		void *x,
		Nd4jLong *xShapeInfo,
		void *y,
		Nd4jLong *yShapeInfo,
		void *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {


	functions::broadcast::BroadcastBool<X, Z>::template transformCuda<OpClass>(
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			NULL,
			tadOnlyShapeInfo,
			tadOffsets,
			tadOnlyShapeInfoZ,
			tadOffsetsZ);
}

template<typename X, typename Z, typename OpClass>
static __global__ void broadcastBoolSimple(
        void *x,
        Nd4jLong *xShapeInfo,
        void *y,
        Nd4jLong *yShapeInfo,
        void *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    broadcastBoolSimpleGeneric<X, Z, OpClass>(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
}

// broadcast kernel call
// DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float, INPUT(float *x, Nd4jLong *xShapeInfo, float *y, Nd4jLong *yShapeInfo, float *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
// DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, double, INPUT(double *x, Nd4jLong *xShapeInfo, double *y, Nd4jLong *yShapeInfo, double *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
// DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float16, INPUT(float16 *x, Nd4jLong *xShapeInfo, float16 *y, Nd4jLong *yShapeInfo, float16 *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))


namespace functions {
    namespace broadcast {

        template<typename X, typename Z>
        template <typename OpClass>
        __host__ void BroadcastBool<X,Z>::intermediateBroadcast(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            broadcastBoolSimple<X, Z, OpClass><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
        }

        template<typename X, typename Y>
        __host__ void BroadcastBool<X,Y>::executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            DISPATCH_BY_OPNUM_TT(intermediateBroadcast,  PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_BOOL_OPS))

	        DEBUG_KERNEL(stream, opNum);
        }



        template<typename X, typename Z>
        template <typename OpType>
		__device__ void BroadcastBool<X,Z>::transformCuda(
		void *vx,
		Nd4jLong *xShapeInfo,
		void *vy,
		Nd4jLong *yShapeInfo,
		void *vresult,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {

      auto x = static_cast<X*>(vx);
      auto y = static_cast<X*>(vy);
      auto result = static_cast<Z*>(vresult);

		//decompose in to several sub tads after
		//moving all dimensions (in sorted order)
		//to the back.
		//permuted version of the x shape info for setting up the tad problem
	  __shared__ Nd4jLong tadLength;
      __shared__ Nd4jLong tadEWS;
      __shared__ int tadRank;
      __shared__ int numTads;
      __shared__ Nd4jLong *tadShape;
      __shared__ Nd4jLong *tadStride;
      __shared__ Nd4jLong yEWS;
      __shared__ Nd4jLong zEWS;
      __shared__ int zRank;
      __shared__ Nd4jLong *zShape;
      __shared__ Nd4jLong *zStride;
      __shared__ int yRank;
      __shared__ Nd4jLong *yShape;
      __shared__ Nd4jLong *yStride;
      if (threadIdx.x == 0) {
        if (tadOnlyShapeInfoZ == nullptr) {
            tadOnlyShapeInfoZ = tadOnlyShapeInfo;
            tadOffsetsZ = tadOffsets;
        }

   	    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        yEWS = shape::elementWiseStride(yShapeInfo);
      	zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);

        if (tadEWS < 1 || zEWS < 1 || yEWS < 1 || dimensionLength > 1) {
            tadRank = shape::rank(tadOnlyShapeInfo);
            tadShape = shape::shapeOf(tadOnlyShapeInfo);
      	    tadStride = shape::stride(tadOnlyShapeInfo);
      	    zRank = shape::rank(tadOnlyShapeInfoZ);
      	    zShape = shape::shapeOf(tadOnlyShapeInfoZ);
      	    zStride = shape::stride(tadOnlyShapeInfoZ);
      	    yRank = shape::rank(yShapeInfo);
      	    yShape = shape::shapeOf(yShapeInfo);
      	    yStride = shape::stride(yShapeInfo);
        }
      }
      __syncthreads();

		for (int r = blockIdx.x; r < numTads; r += gridDim.x) {


            __shared__ Nd4jLong tadOffsetForBlock;
            __shared__ Nd4jLong tadOffsetForBlockZ;
            __shared__ Z *rR;
            __shared__ X *rX;
            if (threadIdx.x == 0) {
                tadOffsetForBlockZ = tadOffsetsZ[r];
                auto vr = reinterpret_cast<void *>(result);
                auto vx = reinterpret_cast<void *>(x);
                if (vr != vx)
                    tadOffsetForBlock = tadOffsets[r];
                else
                    tadOffsetForBlock = tadOffsetForBlockZ;

                rR = result + tadOffsetForBlockZ;
                rX = x + tadOffsetForBlock;
            }
            __syncthreads();


            if(tadEWS > 0 && zEWS > 0 && yEWS > 0 && dimensionLength == 1) {
            	if (tadEWS == 1 && yEWS == 1 && zEWS == 1) {
                	for (int i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i] = OpType::op(rX[i], y[i]);
                	}
                } else {
					for (int i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i * zEWS] = OpType::op(rX[i * tadEWS], y[i * yEWS]);
                	}
                }
            }
            else {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];
                Nd4jLong zCoord[MAX_RANK];

                for (Nd4jLong i = threadIdx.x; i < tadLength; i+= blockDim.x) {

                    if (shape::order(tadOnlyShapeInfo) == 'c') {
                        shape::ind2subC(tadRank,tadShape, i, tadLength, xCoord);
                        shape::ind2subC(yRank, yShape, i, tadLength, yCoord);
                    } else {
                        shape::ind2sub(tadRank,tadShape, i, tadLength, xCoord);
                        shape::ind2sub(yRank, yShape, i, tadLength, yCoord);
                    }

                    if (shape::order(tadOnlyShapeInfoZ) == 'c')
                        shape::ind2subC(zRank,zShape, i, tadLength, zCoord);
                    else
                        shape::ind2sub(zRank,zShape, i, tadLength, zCoord);

                    auto xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);
                    auto zOffset = shape::getOffset(tadOffsetForBlockZ, zShape, zStride, zCoord, zRank);
                    auto yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                    result[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                }
            }
		}
	}

	    BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT BroadcastBool, , LIBND4J_TYPES, BOOL_TYPES);
    }
}