/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <math/templatemath.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <system/pairwise_util.h>
#include <system/dll.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/DebugHelper.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
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
		static __device__ void execScalarCuda(const void *x, const Nd4jLong *xShapeInfo,
                                              const void *y, const Nd4jLong *yShapeInfo,
                                              void *extraParams,
                                              void *z, const Nd4jLong *zShapeInfo,
                                              int *allocationPointer, void *reductionBuffer,
                                              const Nd4jLong *tadOnlyShapeInfo);

		template<typename OpType>
		static __device__ void transformAll(const void *vx, const Nd4jLong *xShapeInfo,
                                            const void *vy, const Nd4jLong *yShapeInfo,
                                            void *extraParams,
                                            void *vz, const Nd4jLong *zShapeInfo,
                                            int *dimension, int dimensionLength,
                                            int postProcessOrNot,
                                            int *allocationPointer,
                                            const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                                            const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets);
		
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
		static __device__ void transform(const void *vx, const Nd4jLong *xShapeInfo,
                                         const void *vy, const Nd4jLong *yShapeInfo,
                                         void *extraParams,
                                         void *vz, const Nd4jLong *zShapeInfo,
                                         int *dimension, int dimensionLength,
                                         int postProcessOrNot,
                                         int *allocationPointer,
                                         const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                         const Nd4jLong *yTadOnlyShapeInfo, const Nd4jLong *yTadOffsets);
		

		static __device__ void execCuda(int opNum,
                                        const void *vx, const Nd4jLong *xShapeInfo,
                                        const void *vy, const Nd4jLong *yShapeInfo,
                                        void *extraParams,
                                        void *vz, const Nd4jLong *zShapeInfo,
                                        int *dimension, int dimensionLength,
                                        int postProcessOrNot,
                                        int *allocationPointer,
                                        const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                        const Nd4jLong *yTadOnlyShapeInfo, const Nd4jLong *yTadOffsets);


		static __device__ void execAllCuda(int opNum,
                                           const void *vx, const Nd4jLong *xShapeInfo,
                                           const void *vy, const Nd4jLong *yShapeInfo,
                                           void *extraParams,
                                           void *vz, const Nd4jLong *zShapeInfo,
                                           int *dimension, int dimensionLength,
                                           int postProcessOrNot,
                                           int *allocationPointer,
                                           const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                           const Nd4jLong *yTadOnlyShapeInfo, const Nd4jLong *yTadOffsets);


		static __device__ void execScalarCuda(int opNum,
                                              const void *vx, const Nd4jLong *xShapeInfo,
                                              const void *vy, const Nd4jLong *yShapeInfo,
                                              void *extraParams,
                                              void *vz, const Nd4jLong *zShapeInfo,
                                              int * allocationPointer, void *reductionBuffer,
                                              const Nd4jLong *tadOnlyShapeInfo);


		static __host__ void exec(dim3 launchDims, cudaStream_t *stream,
                                  int opNum,
                                  const void *vx, const Nd4jLong *xShapeInfo,
                                  const void *vy, const Nd4jLong *yShapeInfo,
                                  void *extraParams,
                                  void *vz, const Nd4jLong *zShapeInfo,
                                  int *dimension, int dimensionLength,
                                  int postProcessOrNot,
                                  int *allocationPointer,
                                  const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                  const Nd4jLong *yTadOnlyShapeInfo, const Nd4jLong *yTadOffsets);

		static __host__ void execAll(dim3 launchDims, cudaStream_t *stream,
                                     int opNum,
                                     const void *vx, const Nd4jLong *xShapeInfo,
                                     const void *vy, const Nd4jLong *yShapeInfo,
                                     void *extraParams,
                                     void *vz, const Nd4jLong *zShapeInfo,
                                     int *dimension, int dimensionLength,
                                     int postProcessOrNot,
                                     int *allocationPointer,
                                     const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                     const Nd4jLong *yTadOnlyShapeInfo, const Nd4jLong *yTadOffsets);

		static __host__ void execScalar(dim3 launchDims, cudaStream_t *stream,
                                        int opNum,
                                        const void *vx, const Nd4jLong *xShapeInfo,
                                        const void *vy, const Nd4jLong *yShapeInfo,
                                        void *extraParams,
                                        void *vz, const Nd4jLong *zShapeInfo,
                                        int* allocationPointer, void *reductionBuffer,
                                        const Nd4jLong *tadOnlyShapeInfo);

#else

		template<typename OpType>
		static void execScalar(const void *vx, const Nd4jLong *xShapeInfo,
		                       void *vextraParams,
                               const void *vy, const Nd4jLong *yShapeInfo,
                               void *vz, const Nd4jLong *zShapeInfo);

		
		static void execScalar(int opNum,
                               const void *x, const Nd4jLong *xShapeInfo,
                               void *extraParamsVals,
                               const void *y, const Nd4jLong *yShapeInfo,
                               void *z, const Nd4jLong *zShapeInfo);

		
		template<typename OpType>
		static void exec(const void *vx, const Nd4jLong *xShapeInfo,
		                 void *vextraParams,
                         const void *vy, const Nd4jLong *yShapeInfo,
                         void *vz, const Nd4jLong *zShapeInfo,
                         int *dimension, int dimensionLength,
                         int64_t start, int64_t stop);

		
		template<typename OpType>
		static void exec(const void *vx, const Nd4jLong *xShapeInfo,
		                 void *vextraParams,
                         const void *vy, const Nd4jLong *yShapeInfo,
                         void *vz, const Nd4jLong *zShapeInfo,
                         int *dimension, int dimensionLength,
                         const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                         int64_t start, int64_t stop);


		template<typename OpType>
		static void execAll(const void *vx, const Nd4jLong *xShapeInfo,
		                    void *vextraParams,
                            const void *vy, const Nd4jLong *yShapeInfo,
                            void *vz, const Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                            const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                            int64_t start, int64_t stop);
		
		
		static void exec(int opNum,
                         const void *vx, const Nd4jLong *xShapeInfo,
                         void *extraParamsVals,
                         const void *vy, const Nd4jLong *yShapeInfo,
                         void *vz, const Nd4jLong *zShapeInfo,
                         int *dimension, int dimensionLength,
                         int64_t start, int64_t stop);


		static void exec(int opNum,
                         const void *vx, const Nd4jLong *xShapeInfo,
                         void *extraParamsVals,
                         const void *vy, const Nd4jLong *yShapeInfo,
                         void *vz, const Nd4jLong *zShapeInfo,
                         int *dimension, int dimensionLength,
                         const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                         int64_t start, int64_t stop);

		
		static void execAll(int opNum,
                            const void *vx, const Nd4jLong *xShapeInfo,
                            void *extraParamsVals,
                            const void *vy, const Nd4jLong *yShapeInfo,
                            void *vz, const Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                            const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets,
                            int64_t start, int64_t stop);
#endif
};



}
}

#ifdef __CUDACC__

#endif



#endif /* REDUCE3_H_ */
