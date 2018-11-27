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

/*
 * reduce3.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE3_H_
#define REDUCE3_H_

#define EXTRA_PARAMS_LENGTH 10

#include <templatemath.h>
#include <helper_cuda.h>
#include <helpers/sharedmem.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pairwise_util.h>
#include <dll.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <ops/ops.h>
#include <op_boilerplate.h>
#include <OmpLaunchHelper.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include "legacy_ops.h"

using namespace simdOps;

namespace functions {
namespace reduce3   {

/**
 * Reduce involving
 * 2 arrays
 */
template<typename X, typename Y>
class Reduce3 {

	public:

#ifdef __CUDACC__
        virtual __device__
		inline Y opAtomic(X d1, X d2, Y *extraParamsRef) = 0;

		/**
			* Aggregate shared memory
		* @param sPartialsRef
		* @param tid
		* @param extraParams
		*/		
		template<typename OpType>
		static __device__ void aggregatePartials(void* sPartials, Nd4jLong tid, Nd4jLong numItems, void *extraParams);
		
		template<typename OpType>
		static __device__ void execScalarCuda(void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, int *allocationPointer, void *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

		template<typename OpType>
		static __device__ void transformAll(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets);
		
		/**
         Perform a reduction
         @param n the number of elements
         @param xOffset the starting offset
         @param dx the data to perform the reduction on
         @param incx the increment on which to perform the reduction
         @param extraParams extra parameters used for calculations
         @param result where to store the result of the reduction
        */
		template<typename OpType>
		static __device__ void transform(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);
		

		static __device__ void execCuda(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);


		static __device__ void execAllCuda( const int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);


		static __device__ void execScalarCuda(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int * allocationPointer, void *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);


		static __host__ void exec(dim3 launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

		static __host__ void execAll(dim3 launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

		static __host__ void execScalar(dim3 launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int* allocationPointer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo);



#endif

		template<typename OpType>
		static void execScalar(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo);

		
		static void execScalar(const int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParamsVals, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo);

		
		template<typename OpType>
		static void exec(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength);

		
		template<typename OpType>
		static void exec(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);


		template<typename OpType>
		static void execAll(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength,  Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets);
		
		
		static void exec(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *extraParamsVals, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength);


		static void exec(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *extraParamsVals, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

		
		static void execAll(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *extraParamsVals, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets);

};



}
}

#ifdef __CUDACC__
/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param zShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post
 */
// template <typename X, typename Y>
// __device__ void reduce3Generic(
// 		const int opNum,
// 		void *dx,
// 		Nd4jLong *xShapeInfo,
// 		void *dy,
// 		Nd4jLong *yShapeInfo,
// 		void *extraParams,
// 		void *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

// 	__shared__ UnifiedSharedMemory *manager;

// 	if (threadIdx.x == 0) {
// 		extern __shared__ unsigned char shmem[];
// 		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
// 		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<X,Y>), sizeof(shape::TAD), shape::rank(xShapeInfo));

// 	}
// 	__syncthreads();

// 	functions::reduce3::Reduce3<X,Y>::exec(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			dimension,
// 			dimensionLength,
// 			postProcessOrNot,
// 			allocationPointer,
// 			manager,
// 			tadOnlyShapeInfo,
// 			tadOffsets,
// 			yTadOnlyShapeInfo,
// 			yTadOffsets);
// }

// template <typename X, typename Y>
// __device__ void reduce3AllGeneric(
// 		const int opNum,
// 		void *dx,
// 		Nd4jLong *xShapeInfo,
// 		void *dy,
// 		Nd4jLong *yShapeInfo,
// 		void *extraParams,
// 		void *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot,
// 		int *allocationPointer,
// 		Nd4jLong *tadOnlyShapeInfo,
// 		Nd4jLong *tadOffsets,
// 		Nd4jLong *yTadOnlyShapeInfo,
// 		Nd4jLong *yTadOffsets) {

// 	__shared__ UnifiedSharedMemory *manager;

// 	if (threadIdx.x == 0) {
// 		extern __shared__ unsigned char shmem[];
// 		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
// 		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<X,Y>), sizeof(shape::TAD), shape::rank(xShapeInfo));

// 	}
// 	__syncthreads();

// 	functions::reduce3::Reduce3<X,Y>::execAllCuda(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			dimension,
// 			dimensionLength,
// 			postProcessOrNot,
// 			allocationPointer,
// 			manager,
// 			tadOnlyShapeInfo,
// 			tadOffsets,
// 			yTadOnlyShapeInfo,
// 			yTadOffsets);
// }

// template <typename X, typename Y>
// __device__ void reduce3ScalarGeneric(
// 		int opNum,
// 		void *dx,
// 		Nd4jLong *xShapeInfo,
// 		void *dy,
// 		Nd4jLong *yShapeInfo,
// 		void *extraParams,
// 		void *result,
// 		Nd4jLong *zShapeInfo, int *allocationPointer,
// 		void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

// 	__shared__ UnifiedSharedMemory *manager;

// 	if (threadIdx.x == 0) {
// 		extern __shared__ unsigned char shmem[];
// 		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
// 		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<X,Y>), sizeof(shape::TAD), shape::rank(xShapeInfo));
// 	}
// 	__syncthreads();

// 	functions::reduce3::Reduce3<X,Y>::execScalarCuda(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			allocationPointer,
// 			reductionBuffer,
// 			manager,
// 			tadOnlyShapeInfo);
// }

// /**
//  * The driver api
//  * @param opNum the number
//  * @param n the length of the reduce
//  * @param dx the input data
//  * @param xShapeInfo the shape information
//  * @param dy the pair wise reduce
//  * @param yShapeInfo the shape information for y
//  * @param extraParams the extra parameters in the operation
//  * @param result where to store the result
//  * @param zShapeInfo the shape information
//  * @param dimension the dimension to reduce along long
//  * @param dimensionLength the dimension length
//  * @param postProcessOrNot whether to post [
//  */
// extern "C"
// __global__ void reduce3Double(
// 		int opNum,
// 		double *dx,
// 		Nd4jLong *xShapeInfo,
// 		double *dy,
// 		Nd4jLong *yShapeInfo,
// 		double *extraParams,
// 		double *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3Generic<double, double>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			dimension,
// 			dimensionLength,
// 			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
// }

// extern "C"
// __global__ void reduce3AllDouble(
// 		int opNum,
// 		double *dx,
// 		Nd4jLong *xShapeInfo,
// 		double *dy,
// 		Nd4jLong *yShapeInfo,
// 		double *extraParams,
// 		double *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3AllGeneric<double, double>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			dimension,
// 			dimensionLength,
// 			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

// }

// *
//  * The driver api
//  * @param opNum the number
//  * @param n the length of the reduce
//  * @param dx the input data
//  * @param xShapeInfo the shape information
//  * @param dy the pair wise reduce
//  * @param yShapeInfo the shape information for y
//  * @param extraParams the extra parameters in the operation
//  * @param result where to store the result
//  * @param zShapeInfo the shape information
//  * @param gpuInformation the gpu information
//  * @param dimension the dimension to reduce along long
//  * @param dimensionLength the dimension length
//  * @param postProcessOrNot whether to post [
 
// extern "C"
// __global__ void reduce3Float(
// 		int opNum,
// 		float *dx,
// 		Nd4jLong *xShapeInfo,
// 		float *dy,
// 		Nd4jLong *yShapeInfo,
// 		float *extraParams,
// 		float *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3Generic<float,float>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			dimension,
// 			dimensionLength,
// 			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

// }

// extern "C"
// __global__ void reduce3AllFloat(
// 		int opNum,
// 		float *dx,
// 		Nd4jLong *xShapeInfo,
// 		float *dy,
// 		Nd4jLong *yShapeInfo,
// 		float *extraParams,
// 		float *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3AllGeneric<float,float>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			dimension,
// 			dimensionLength,
// 			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

// }

// extern "C"
// __global__ void reduce3Half(
// 		int opNum,
// 		float16 *dx,
// 		Nd4jLong *xShapeInfo,
// 		float16 *dy,
// 		Nd4jLong *yShapeInfo,
// 		float16 *extraParams,
// 		float16 *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3Generic<float16,float16>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			dimension,
// 			dimensionLength,
// 			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

// }

// extern "C"
// __global__ void reduce3AllHalf(
// 		int opNum,
// 		float16 *dx,
// 		Nd4jLong *xShapeInfo,
// 		float16 *dy,
// 		Nd4jLong *yShapeInfo,
// 		float16 *extraParams,
// 		float16 *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3AllGeneric<float16,float16>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo,
// 			dimension,
// 			dimensionLength,
// 			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

// }

// extern "C"
// __global__ void reduce3ScalarFloat(
// 		int opNum,
// 		float *dx,
// 		Nd4jLong *xShapeInfo,
// 		float *dy,
// 		Nd4jLong *yShapeInfo,
// 		float *extraParams,
// 		float *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, float *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3ScalarGeneric<float,float>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo, allocationPointer,
// 			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

// }

// extern "C" __global__ void reduce3ScalarHalf(
// 		int opNum,
// 		float16 *dx,
// 		Nd4jLong *xShapeInfo,
// 		float16 *dy,
// 		Nd4jLong *yShapeInfo,
// 		float16 *extraParams,
// 		float16 *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, float16 *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3ScalarGeneric<float16,float16>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo, allocationPointer,
// 			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

// }

// extern "C"
// __global__ void reduce3ScalarDouble(
// 		int opNum,
// 		double *dx,
// 		Nd4jLong *xShapeInfo,
// 		double *dy,
// 		Nd4jLong *yShapeInfo,
// 		double *extraParams,
// 		double *result,
// 		Nd4jLong *zShapeInfo,
// 		int *dimension,
// 		int dimensionLength,
// 		int postProcessOrNot, int *allocationPointer, double *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
// 	reduce3ScalarGeneric<double,double>(
// 			opNum,
// 			dx,
// 			xShapeInfo,
// 			dy,
// 			yShapeInfo,
// 			extraParams,
// 			result,
// 			zShapeInfo, allocationPointer,
// 			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

// }

#endif



#endif /* REDUCE3_H_ */
