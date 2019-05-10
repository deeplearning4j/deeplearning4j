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
// @author raver119@gmail.com
//

#include <Environment.h>
#include <loops/transform_bool.h>
#include <types/types.h>
#include <op_boilerplate.h>

#include <loops/legacy_ops.h>
#include <helpers/DebugHelper.h>

using namespace simdOps;


template <typename X, typename Z, typename OpType>
__global__ void transformBoolSimple(void *dy, Nd4jLong *xShapeInfo, int xRank,
								void *params,
								void *result, Nd4jLong *zShapeInfo, int zRank,
								int *allocationPointer,
								void *reductionPointer,
								Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	functions::transform::TransformBool<X,Z>::template transformCuda<OpType>(dy,xShapeInfo,params,result,zShapeInfo,allocationPointer,reductionPointer,tadShapeInfo, tadOffsets);
}


namespace functions {
    namespace transform {

        template<typename X, typename Y>
        _CUDA_H void TransformBool<X,Y>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShape, int xRank, void *extraParams, void *z, Nd4jLong *zShape, int zRank, int *allocationPointer, void *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			DISPATCH_BY_OPNUM_TT(intermediateShaped, PARAMS(launchDims, stream, x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), TRANSFORM_BOOL_OPS);

            DEBUG_KERNEL(stream, opNum);
        }


        template<typename X, typename Z>
        template <typename OpType>
        __device__ void TransformBool<X,Z>::transformCuda(
						void *vdy,
						Nd4jLong *shapeInfo,
						void *vparams,
						void *vresult,
						Nd4jLong *zShapeInfo,
						int *allocationPointer, void *vreductionPointer, 
						Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

        	auto dy = static_cast<X*>(vdy);
		    auto result = static_cast<Z*>(vresult);
		    auto params = static_cast<X*>(vparams);
		    auto reductionPointer = static_cast<Z*>(vreductionPointer);

		    if(OpType::requiresSpecial) {
			    OpType::execSpecialCuda(dy,shapeInfo,result,zShapeInfo,params, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets);
			    return;
		    } else {

		        auto xOrder = shape::order(shapeInfo);
		        auto zOrder = shape::order(zShapeInfo);

		        auto xEws = shape::elementWiseStride(shapeInfo);
    		    auto zEws = shape::elementWiseStride(zShapeInfo);
	    	    auto tid = blockIdx.x * blockDim.x + threadIdx.x;

                __shared__ Nd4jLong length;
		        if(threadIdx.x == 0)
			        length = shape::length(shapeInfo);
		        __syncthreads();

				int totalThreads = gridDim.x * blockDim.x;

		        if(xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
					if(xEws == 1 && zEws == 1) {
						/* equal, positive, non-unit increments. */
						for (Nd4jLong i = tid; i < length; i += totalThreads) {
							result[i] = OpType::op(dy[i], params);
						}
					}
					else {
						for (Nd4jLong i = tid; i < length; i += totalThreads) {
							result[i * zEws] = OpType::op(dy[i * xEws], params);
						}
					}
		        }
		        else {
			
		    	    for (Nd4jLong i = tid; i < length; i+= totalThreads) {
						auto xOffset2 = shape::getIndexOffset(i, shapeInfo,  length);
						auto zOffset2 = shape::getIndexOffset(i, zShapeInfo, length);						
	    			    result[zOffset2] = OpType::op(dy[xOffset2], params);
		    	    }
		        }
	        }
	    };


		template<typename X, typename Z>
		template <typename OpType>
		_CUDA_H void TransformBool<X,Z>::intermediateShaped(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShape, int xRank, void *extraParams, void *z, Nd4jLong *zShape, int zRank, int *allocationPointer, void *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			transformBoolSimple<X, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets);
            nd4j::DebugHelper::checkErrorCode(stream, "transformBool(...) failed");
		}

        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT TransformBool, , LIBND4J_TYPES, BOOL_TYPES);
    }
}
