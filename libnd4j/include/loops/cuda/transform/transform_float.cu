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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <Environment.h>
#include <loops/transform_float.h>
#include <op_boilerplate.h>

#include <loops/legacy_ops.h>
#include <helpers/DebugHelper.h>

using namespace simdOps;

template<typename X, typename Z>
__device__ void transformGeneric(
		int opNum,
		Nd4jLong n,
		void *dy,
		Nd4jLong incy,
		void *params,
		void *result,
		Nd4jLong resultStride, int *allocationPointer, void *reductionPointer) {

	functions::transform::TransformFloat<X,Z>::transformCuda(
		opNum,
		n,
		dy,
		incy,
		params,
		result,
		resultStride,
		allocationPointer,
		reductionPointer,
		nullptr);
}

template<typename X, typename Z, typename OpClass>
__device__ void transformSimpleGeneric(
		Nd4jLong n,
		void *dy,
		Nd4jLong incy,
		void *params,
		void *result,
		Nd4jLong resultStride, int *allocationPointer, void *reductionPointer) {

	functions::transform::TransformFloat<X,Z>::template transformCuda<OpClass>(
		n,
		dy,
		incy,
		params,
		result,
		resultStride,
		allocationPointer,
		reductionPointer,
		nullptr);
}



template<typename X, typename Z>
__device__ void transformGeneric(
		int opNum,
		void *dy,
		Nd4jLong *xShapeInfo, int xRank,
		void *params,
		void *result,Nd4jLong *resultShapeInfo, int zRank, int *allocationPointer, void *reductionPointer) {

	functions::transform::TransformFloat<X,Z>::transformCuda(
	    opNum,
	    dy,
	    xShapeInfo,
	    params,
	    result,
	    resultShapeInfo,
	    allocationPointer,
	    reductionPointer,
	    nullptr);


}

template<typename X, typename Z, typename OpClass>
__device__ void transformSimpleGeneric(
		void *dy,
		Nd4jLong *xShapeInfo, int xRank,
		void *params,
		void *result, Nd4jLong *resultShapeInfo, int zRank, int *allocationPointer, void *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::transform::TransformFloat<X,Z>), sizeof(shape::TAD), xRank);
	}
	__syncthreads();
	
    functions::transform::TransformFloat<X,Z>::template transformCuda<OpClass>(
	    dy,
	    xShapeInfo,
	    params,
	    result,
	    resultShapeInfo,
	    allocationPointer,
	    reductionPointer,
		manager, tadShapeInfo, tadOffsets);
}

// transform strided
// DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, float, INPUT(Nd4jLong n, float *x, Nd4jLong xStride, float *extraParams, float *z, Nd4jLong zStride, int *allocationPointer, float *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
// DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, double, INPUT(Nd4jLong n, double *x, Nd4jLong xStride, double *extraParams, double *z, Nd4jLong zStride, int *allocationPointer, double *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
// DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, float16, INPUT(Nd4jLong n, float16 *x, Nd4jLong xStride, float16 *extraParams, float16 *z, Nd4jLong zStride, int *allocationPointer, float16 *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

// transform shaped
// DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, float, INPUT(float *x, Nd4jLong *xShape, int xRank, float *extraParams, float *z, Nd4jLong *zShape, int zRank, int *allocationPointer, float *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
// DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, double, INPUT(double *x, Nd4jLong *xShape, int xRank, double *extraParams, double *z, Nd4jLong *zShape, int zRank, int *allocationPointer, double *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
// DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, float16, INPUT(float16 *x, Nd4jLong *xShape, int xRank, float16 *extraParams, float16 *z, Nd4jLong *zShape, int zRank, int *allocationPointer, float16 *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))



namespace functions {
    namespace transform {

        template<typename X, typename Z>
        _CUDA_H void TransformFloat<X,Z>::executeTransformStrided(dim3 launchDims, cudaStream_t *stream, int opNum, Nd4jLong n, void *x, Nd4jLong xStride, void *extraParams, void *z, Nd4jLong zStride, int *allocationPointer, void *reductionPointer) {
            // DISPATCH_SIMPLE(transformStrided, float, PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

            DEBUG_KERNEL(stream, opNum);
        };

        // template <>
        // _CUDA_H void Transform<double>::executeTransformStrided(dim3 launchDims, cudaStream_t *stream, int opNum, Nd4jLong n, double *x, Nd4jLong xStride, double *extraParams, double *z, Nd4jLong zStride, int *allocationPointer, double *reductionPointer) {
        //     DISPATCH_SIMPLE(transformStrided, double, PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

        //     DEBUG_KERNEL(stream, opNum);
        // };

        // template <>
        // _CUDA_H void Transform<float16>::executeTransformStrided(dim3 launchDims, cudaStream_t *stream, int opNum, Nd4jLong n, float16 *x, Nd4jLong xStride, float16 *extraParams, float16 *z, Nd4jLong zStride, int *allocationPointer, float16 *reductionPointer) {
        //     DISPATCH_SIMPLE(transformStrided, float16, PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

        //     DEBUG_KERNEL(stream, opNum);
        // };

        template<typename X, typename Z>
        _CUDA_H void TransformFloat<X,Z>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShape, int xRank, void *extraParams, void *z, Nd4jLong *zShape, int zRank, int *allocationPointer, void *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
            
            // DISPATCH_SIMPLE(transformShaped, float, PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        // template <>
        // _CUDA_H void Transform<float16>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, float16 *x, Nd4jLong *xShape, int xRank, float16 *extraParams, float16 *z, Nd4jLong *zShape, int zRank, int *allocationPointer, float16 *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
            
        //     DISPATCH_SIMPLE(transformShaped, float16, PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
            
        //     if (nd4j::Environment::getInstance()->isDebug())
		      //   checkCudaErrors(cudaStreamSynchronize(*stream));
        // }

        // template <>
        // _CUDA_H void Transform<double>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, double *x, Nd4jLong *xShape, int xRank, double *extraParams, double *z, Nd4jLong *zShape, int zRank, int *allocationPointer, double *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
            
        //     DISPATCH_SIMPLE(transformShaped, double, PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))

        //     DEBUG_KERNEL(stream, opNum);
        // }

        template<typename X, typename Z>
        template <typename OpType>
        __device__ void TransformFloat<X,Z>::transformCuda(
			void *vdy,
			Nd4jLong *shapeInfo,
			void *vparams,
			void *vresult,
			Nd4jLong *resultShapeInfo,
			int *allocationPointer, void *vreductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

        	auto dy = static_cast<X*>(vdy);
		    auto result = static_cast<Z*>(vresult);
		    auto params = static_cast<Z*>(vparams);
		    auto reductionPointer = static_cast<Z*>(vreductionPointer);

		    if(OpType::requiresSpecial) {
			    OpType::execSpecialCuda(dy,shapeInfo,result,resultShapeInfo,params, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets);
			    return;
		    } else {

    		    auto xShape = shape::shapeOf(shapeInfo);
	    	    auto xStride = shape::stride(shapeInfo);
		        auto xOrder = shape::order(shapeInfo);
		        auto resultOrder = shape::order(resultShapeInfo);
    		    auto xRank = shape::rank(shapeInfo);

		        auto xElementWiseStride = shape::elementWiseStride(shapeInfo);
    		    auto resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
	    	    auto tid = blockIdx.x * blockDim.x + threadIdx.x;

                __shared__ Nd4jLong length;
		        if(threadIdx.x == 0)
			        length = shape::length(shapeInfo);
		        __syncthreads();

		        if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && xOrder == resultOrder) {
			        transformCuda<OpType>(
				    	length,
				    	dy,
				    	xElementWiseStride,
				    	params,
				    	result,
				    	resultElementWiseStride, allocationPointer, reductionPointer, manager);
		        }
		        else {
			        Nd4jLong xCoord[MAX_RANK];
			
		    	    for (Nd4jLong i = tid; i < length; i+= gridDim.x * blockDim.x) {
						shape::ind2sub(xRank,shape::shapeOf(shapeInfo),i, length, xCoord);
						
				        auto xOffset2 = shape::getOffset(0, xShape, xStride, xCoord, xRank);
						auto resultOffset2 = shape::getOffset(0,xShape,shape::stride(resultShapeInfo),xCoord,xRank);
						
	    			    result[resultOffset2] = OpType::op(dy[xOffset2], params);
		    	    }
		        }
	        }
	    };

        template<typename X, typename Z>
        template <typename OpType>
	    __device__ void TransformFloat<X,Z>::transformCuda(
			Nd4jLong n,
			void *vdy,
			Nd4jLong incy,
			void *vparams,
			void *vresult,
			Nd4jLong resultStride,
			int *allocationPointer, void *vreductionPointer, UnifiedSharedMemory *manager) {
		
        	auto dy = static_cast<X*>(vdy);
		    auto result = static_cast<Z*>(vresult);
		    auto params = static_cast<Z*>(vparams);
		    auto reductionPointer = static_cast<Z*>(vreductionPointer);

            int totalThreads = gridDim.x * blockDim.x;
		    Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x;

    		if(incy == 1 && resultStride == 1) {
	    		/* equal, positive, non-unit increments. */
			    for (; i < n; i += totalThreads) {
				    result[i] = OpType::op(dy[i], params);
			    }
		    }
		    else {
			    for (; i < n; i += totalThreads) {
				    result[i * resultStride] = OpType::op(dy[i * incy], params);
			    }
		    }
	    }


        template<typename X, typename Z>
        __device__ void TransformFloat<X,Z>::transformCuda(
			const int opNum,
			void *dy,
			Nd4jLong *shapeInfo,
			void *params,
			void *result,
			Nd4jLong *resultShapeInfo,
			int *allocationPointer,
			void *reductionPointer,
			UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
                // DISPATCH_BY_OPNUM(transformCuda, PARAMS(dy, shapeInfo, params, result, resultShapeInfo, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
	    }

        template<typename X, typename Z>
        __device__ void TransformFloat<X,Z>::transformCuda(
			const int opNum,
			Nd4jLong n,
			void *dy,
			Nd4jLong incy,
			void *params,
			void *result,
			Nd4jLong resultStride,
			int *allocationPointer,
			void *reductionPointer,
			UnifiedSharedMemory *manager) {
                // DISPATCH_BY_OPNUM(transformCuda, PARAMS(n, dy, incy, params, result, resultStride, allocationPointer, reductionPointer, manager), TRANSFORM_OPS);
	    }


        //template class ND4J_EXPORT Transform<float>;
        //template class ND4J_EXPORT Transform<float16>;
        //template class ND4J_EXPORT Transform<double>;

        // BUILD_CALL_1(template __device__ void Transform<float>::transformCuda, float, (float*, Nd4jLong*, float*, float*,Nd4jLong*, int*,float*, UnifiedSharedMemory*, Nd4jLong*, Nd4jLong*), TRANSFORM_OPS)
        // BUILD_CALL_1(template __device__ void Transform<float16>::transformCuda, float16, (float16*, Nd4jLong*, float16*, float16*,Nd4jLong*, int*, float16*, UnifiedSharedMemory*, Nd4jLong*, Nd4jLong*), TRANSFORM_OPS)
        // BUILD_CALL_1(template __device__ void Transform<double>::transformCuda, double, (double*, Nd4jLong*, double*, double*,Nd4jLong*, int*, double*, UnifiedSharedMemory*, Nd4jLong*, Nd4jLong*), TRANSFORM_OPS)
    }
}
