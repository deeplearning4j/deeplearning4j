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
		static __device__ void execScalarCuda(void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, int *allocationPointer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo);

		template<typename OpType>
		static __device__ void transformAll(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets);
		
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
		static __device__ void transform(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);
		

		static __device__ void execCuda(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);


		static __device__ void execAllCuda( const int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);


		static __device__ void execScalarCuda(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *extraParams, void *vz, Nd4jLong *zShapeInfo, int * allocationPointer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo);


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

#endif



#endif /* REDUCE3_H_ */
