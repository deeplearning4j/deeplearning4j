/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXREDUCE_H_
#define INDEXREDUCE_H_
#include "../helpers/shape.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <dll.h>
#include <ops/ops.h>
#include <op_boilerplate.h>
#include <OmpLaunchHelper.h>
#include <helpers/DebugHelper.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include <helpers/TAD.h>


#include "../pairwise_util.h"

#include "legacy_ops.h"

namespace functions {
	namespace indexreduce {

		template<typename T>
		class IndexReduce {
		public:
#ifdef __CUDACC__

	static __device__ void transform(const int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, Nd4jLong *result, Nd4jLong *resultShapeInfo, int *dimension,int dimensionLength, int postProcessOrNot, int *allocationBuffer, void *reductionBuffer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset);

    template<typename OpType>
	static __device__ void aggregatePartials(IndexValue<T> **sPartialsRef, Nd4jLong tid, Nd4jLong numElements,void *extraParams);


    template<typename OpType>
	static __device__ void transform(void *dx, Nd4jLong *xShapeInfo, void *extraParams, Nd4jLong *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);


    static _CUDA_H void executeIndexReduceScalar(dim3 launchDims, cudaStream_t *stream, const int op, void *dx, Nd4jLong *xShapeInfo, int xRank, void *extraParams, Nd4jLong *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);

    static _CUDA_H void executeIndexReduce(dim3 launchDims, cudaStream_t *stream, const int op, void *dx, Nd4jLong *xShapeInfo, int xRank, void *extraParams, Nd4jLong *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);
#endif

		static Nd4jLong execScalar(const int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams);

		static void exec(const int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, Nd4jLong *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset);

		template<typename OpType>
		static _CUDA_H Nd4jLong execScalar(void *x, Nd4jLong *xShapeInfo, void *extraParams);

		template<typename OpType>
		static _CUDA_H void exec(void *x, Nd4jLong *xShapeInfo, void *extraParams, Nd4jLong *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset);
		};
	}
}

#endif /* INDEXREDUCE_H_ */

