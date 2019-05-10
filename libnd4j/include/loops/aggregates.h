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

#ifndef LIBND4J_AGGREGATES_H
#define LIBND4J_AGGREGATES_H

#include <ops/aggregate_ops.h>
#include <helpers/DebugHelper.h>
#include <helpers/helper_ptrmap.h>

namespace functions {
namespace aggregate {

        template<typename X>
        class AggregatedFunction {

        public:
#ifdef __CUDACC__
            template<typename OpClass>
            __device__ static void execCuda(X **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  X *realArguments, int numRealArguments);

            __device__ static void execCuda(int opNum, X **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  X *realArguments, int numRealArguments);
      
            __device__ static void aggregateBatch(int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments);

            __host__ static void aggregateBatchKernelGeneric(dim3& launchDims, cudaStream_t *stream, int opNum, int numAggregates, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments);

            __host__ static void aggregateKernelGeneric(dim3& launchDims, cudaStream_t *stream, int opNum, void **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, void *realArguments, int numRealArguments);
            
#endif

             template<typename OpClass>
            inline static void exec(X **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  X *realArguments, int numRealArguments) {
                OpClass::executeAggregate(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
            }

            inline static void exec(int opNum, X **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, X *realArguments, int numRealArguments) {
                DISPATCH_BY_OPNUM_T(exec, PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), AGGREGATE_OPS);
            }
		};
}
}

#ifdef __CUDACC__


#endif

#endif //LIBND4J_AGGREGATES_H
