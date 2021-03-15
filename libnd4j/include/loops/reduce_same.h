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


#ifndef REDUCE_SAME_H
#define REDUCE_SAME_H
#include <system/dll.h>
//#include <string>
#include <stdio.h>
#include <helpers/shape.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math/templatemath.h>
#include <system/nd4jmalloc.h>
#include <system/pairwise_util.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>
#include <memory/Workspace.h>

#pragma once
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "legacy_ops.h"

//an op for the kernel
namespace functions {
    namespace reduce {

/**
 * A reduce function
 * reduces a vector down to
 * a subset of itself
 * via aggregating member
 * elements.
 */
        template<typename X>
        class ReduceSameFunction {
        public:
#ifdef __CUDACC__

            template<typename OpType>
            static __device__ void aggregatePartials(void *sPartials, Nd4jLong tid, Nd4jLong numItems, void *extraParams);

            template<typename OpType>
            static __device__ void execScalarCuda( void const* vx, Nd4jLong const *xShapeInfo, void *extraParams, void *vz, Nd4jLong const* zShapeInfo, void *reductionBuffer, Nd4jLong const* tadOnlyShapeInfo);

            static __device__ void execScalarCudaLegacy(int opNum, void const* vx, Nd4jLong const* xShapeInfo, void *extraParams, void *vz, Nd4jLong const* zShapeInfo, void *reductionBuffer, Nd4jLong const* tadOnlyShapeInfo);

            template<typename OpType>
            static __device__ void transformCudaXD(const void *vx, const Nd4jLong *outerXTadShapeInfo, const Nd4jLong *innerXTadShapeInfo, void *extraParams, void* reductionBuffer, void *vz, const Nd4jLong *zShapeInfo);

            template<typename OpType>
            static __host__ void intermediateScalar(dim3 launchDims, cudaStream_t *stream, void const* vx, Nd4jLong const* xShapeInfo, Nd4jLong const* hXShapeInfo, void *extraParams, void *vz, Nd4jLong const* zShapeInfo, Nd4jLong const* hZShapeInfo, int *dimension, int dimensionLength, void *reductionBuffer, Nd4jLong const* tadOnlyShapeInfo);

            template<typename OpType>
            static __host__ void intermediateXD(dim3 launchDims, cudaStream_t *stream, const void *vx, const Nd4jLong *dXShapeInfo, const Nd4jLong *hXShapeInfo, void *extraParams, void* reductionBuffer, void *vz, const Nd4jLong *dZShapeInfo, const Nd4jLong *hZShapeInfo, const int* dims);

            static __host__ void execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, void const* vx, Nd4jLong const* xShapeInfo, Nd4jLong const*  hXShapeInfo, void *extraParams, void *vz, Nd4jLong const* zShapeInfo, Nd4jLong const* hZShapeInfo, int *dimension, int dimensionLength, void *reductionBuffer, Nd4jLong const* tadOnlyShapeInfo);

            static __host__ void execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, const void *vx, const Nd4jLong *dXShapeInfo, const Nd4jLong *hXShapeInfo, void *extraParams, void* reductionBuffer, void *vz, const Nd4jLong *dZShapeInfo, const Nd4jLong *hZShapeInfo, const int *dims);
#else

            /**
             * Reduce down to 1 number
             * @param x the input
             * @param xShapeInfo the shape information
             * for the input
             * @param extraParams the extra params
             * @return
             */
            template<typename OpType>
            static _CUDA_H X execScalar(const void *x, const Nd4jLong *xShapeInfo,
                                        void *extraParams);

            template<typename OpType>
            static _CUDA_H void execScalar(const void *x, const Nd4jLong *xShapeInfo,
                                           void *extraParams,
                                           void *z, const Nd4jLong *zShapeInfo);


            static X execScalar(int opNum,
                                const void *x, const Nd4jLong *xShapeInfo,
                                void *extraParams);

            static void execScalar(int opNum,
                                   const void *x, const Nd4jLong *xShapeInfo,
                                   void *extraParams,
                                   void *z, const Nd4jLong *zShapeInfo);

            static void exec(int opNum, sd::memory::Workspace* workspace,
                             const void *vx, const Nd4jLong *xShapeInfo,
                             void *vextraParams,
                             void *vz, const Nd4jLong *zShapeInfo,
                             const int *dims);

            /**
             * Execute on the cpu
             * @param x the input data
             * @param xShapeInfo the shape information for x
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfoBuffer the shape information
             * @param dimension the dimension to perform
             * the reduce along long
             * @param dimensionLength the length of the dimension buffer
             */

            template<typename OpType>
            static void _CUDA_H exec(sd::memory::Workspace* workspace,
                                    const void *vx, const Nd4jLong *xShapeInfo,
                                    void *vextraParams,
                                    void *vz, const Nd4jLong *zShapeInfo,
                                    const int *dims);

            /**
            * CPU implementation
            * @param x the input data
            * @param xShapeInfo the shape information for
            * the input data
            * @param extraParams the extra parameters for the problem
            * @param result the result buffer
            * @param resultShapeInfo the shape information
            */
            template<typename OpType>
            static void _CUDA_H exec(const void *x, const Nd4jLong *xShapeInfo,
                                     void *extraParams,
                                     void *result, const Nd4jLong *resultShapeInfo);



            /**
            * Reduce down to 1 number
            * @param x the input
            * @param xShapeInfo the shape information
            * for the input
            * @param extraParams the extra params
            * @return
            */
            template<typename OpType>
            static X _CUDA_H execScalar(const void *x, Nd4jLong xElementWiseStride,
                                        Nd4jLong length,
                                        void *extraParams);

#endif
        };

#ifdef __CUDACC__
        /**
    *
    * @param extraParams
    * @param sPartials
    * @param sMemSize
    */
        template<typename T>
        __device__ void initializeShared(T *extraParams, T **sPartials, int sMemSize);

#endif

    }

}

#endif

