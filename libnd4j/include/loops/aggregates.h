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
#include <helpers/helper_ptrmap.h>

#include "legacy_ops.h"

namespace functions {
    namespace aggregate {

        template<typename T>
        class AggregatedFunction {

        public:
#ifdef __CUDACC__
            template<typename OpClass>
            __device__ inline static void execCuda(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  T *realArguments, int numRealArguments) {
                OpClass::executeAggregateCuda(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
            }
#endif

            template<typename OpClass>
            inline static void exec(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  T *realArguments, int numRealArguments) {
                OpClass::executeAggregate(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
            }

            inline static void exec(int opNum, T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
                DISPATCH_BY_OPNUM(exec, PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), AGGREGATE_OPS);
            }
		};
    }
}

#ifdef __CUDACC__

template <typename T, typename OpClass>
__device__ void aggregateGeneric(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
    functions::aggregate::AggregatedFunction<T>:: template execCuda<OpClass>(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
};


template <typename T, typename OpClass>
__device__ void aggregateBatchGeneric(int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments) {

    nd4j::PointersHelper<T> helper(ptrToArguments, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

    // TODO: we probably should lift this restriction
    __shared__ int *intArrays[32];

    __shared__ T **arguments;
    __shared__ Nd4jLong **shapes;
    __shared__ int *idxArg;
    __shared__ T *realArg;

    for(int r = blockIdx.x; r < numAggregates; r += gridDim.x) {
        if (threadIdx.x == 0) {
            arguments = helper.getArguments(r);
            shapes = helper.getShapeArguments(r);
            idxArg = helper.getIndexArguments(r);
            realArg = helper.getRealArguments(r);
        }

        // we fill intArrays param in parallel within block
        if (threadIdx.x < 32 && threadIdx.x < maxIntArrays) {
            intArrays[threadIdx.x] = helper.getIntArrayArguments(r, threadIdx.x);
        }
        __syncthreads();

        functions::aggregate::AggregatedFunction<T>::template execCuda<OpClass>(arguments, helper.getNumArguments(r), shapes, helper.getNumShapeArguments(r), idxArg, helper.getNumIndexArguments(r), intArrays, helper.getNumIntArrayArguments(r), realArg, helper.getNumRealArguments(r));
    }
};

// simple aggregates
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, float, INPUT(float **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, double, INPUT(double **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, double *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, float16, INPUT(float16 **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float16 *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))


// batched aggregates
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, float, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, float16, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, double, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

#endif

#endif //LIBND4J_AGGREGATES_H
