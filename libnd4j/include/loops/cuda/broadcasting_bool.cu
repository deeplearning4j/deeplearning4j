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
static __global__ void broadcastBoolSimple(
        void *x,
        Nd4jLong *xShapeInfo,
        void *y,
        Nd4jLong *yShapeInfo,
        void *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    
    functions::broadcast::BroadcastBool<X, Z>::template transformCuda<OpClass>(x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,tadOnlyShapeInfo,tadOffsets,tadOnlyShapeInfoZ,tadOffsetsZ);
}

namespace functions {
    namespace broadcast {

        template<typename X, typename Z>
        template <typename OpClass>
        __host__ void BroadcastBool<X,Z>::intermediateBroadcast(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            broadcastBoolSimple<X, Z, OpClass><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
            nd4j::DebugHelper::checkErrorCode(stream, "intermediateBroadcastBool(...) failed");
        }

        template<typename X, typename Y>
        __host__ void BroadcastBool<X,Y>::execBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
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
		int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {

      auto x = reinterpret_cast<X*>(vx);
      auto y = reinterpret_cast<X*>(vy);
      auto result = reinterpret_cast<Z*>(vresult);

		//decompose in to several sub tads after
		//moving all dimensions (in sorted order)
		//to the back.
		//permuted version of the x shape info for setting up the tad problem
	  __shared__ Nd4jLong tadLength;
      __shared__ Nd4jLong tadLengthZ;
      __shared__ Nd4jLong tadEWS;
            __shared__ int numTads;
      __shared__ Nd4jLong yEWS;
      __shared__ Nd4jLong zEWS;
      
      if (threadIdx.x == 0) {
        if (tadOnlyShapeInfoZ == nullptr) {
            tadOnlyShapeInfoZ = tadOnlyShapeInfo;
            tadOffsetsZ = tadOffsets;
        }

   	    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        tadLengthZ = shape::length(tadOnlyShapeInfoZ);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        yEWS = shape::elementWiseStride(yShapeInfo);
      	zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);
      
      }
      __syncthreads();

        __shared__ Nd4jLong tadOffsetForBlock;
        __shared__ Nd4jLong tadOffsetForBlockZ;
        __shared__ Z *rR;
        __shared__ X *rX;

		for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
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
                for (Nd4jLong i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    auto xOffset = tadOffsetForBlock  + shape::getIndexOffset(i, tadOnlyShapeInfo,  tadLength);
                    auto zOffset = tadOffsetForBlockZ + shape::getIndexOffset(i, tadOnlyShapeInfoZ, tadLengthZ);
                    auto yOffset = shape::getIndexOffset(i, yShapeInfo, tadLength);


                    result[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                }
            }
		}
	}

	    BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT BroadcastBool, , LIBND4J_TYPES, BOOL_TYPES);
    }
}