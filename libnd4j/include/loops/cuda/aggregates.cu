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

//
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 27.11.2018
//

#include "../aggregates.h"

namespace functions {
namespace aggregate {

///////////////////////////////////////////////////////////////////////
template <typename X>
template<typename OpClass>
__device__ void AggregatedFunction<X>::execCuda(X **arguments, int numArguments, 
                                        Nd4jLong **shapeArguments, int numShapeArguments, 
                                        int *indexArguments, int numIndexArguments, 
                                        int **intArrays, int numIntArrays,  
                                        X *realArguments, int numRealArguments) {

    OpClass::executeAggregateCuda(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
}

///////////////////////////////////////////////////////////////////////
template <typename X>
__device__ void AggregatedFunction<X>::execCuda(int opNum, 
                                        X **arguments, int numArguments, 
                                        Nd4jLong **shapeArguments, int numShapeArguments, 
                                        int *indexArguments, int numIndexArguments, 
                                        int **intArrays, int numIntArrays,  
                                        X *realArguments, int numRealArguments) {
    
    DISPATCH_BY_OPNUM_T(execCuda, PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), AGGREGATE_OPS);
}

///////////////////////////////////////////////////////////////////////
template <typename X>
__global__ static void execAggregateKernel(int opNum,
                                void **varguments, int numArguments,
                                Nd4jLong **shapeArguments, int numShapeArguments,
                                int *indexArguments, int numIndexArguments,
                                int **intArrays, int numIntArrays,
                                void *vrealArguments, int numRealArguments) {

    auto arguments = reinterpret_cast<X**>(varguments);
    auto realArguments = reinterpret_cast<X*>(vrealArguments);
    functions::aggregate::AggregatedFunction<X>::execCuda(opNum, arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);    
}

///////////////////////////////////////////////////////////////////////
template <typename X>
__host__ void AggregatedFunction<X>::aggregateKernelGeneric(dim3& launchDims, cudaStream_t *stream,
                                int opNum,
                                void **arguments, int numArguments,
                                Nd4jLong **shapeArguments, int numShapeArguments,
                                int *indexArguments, int numIndexArguments,
                                int **intArrays, int numIntArrays,
                                void *realArguments, int numRealArguments) {

    execAggregateKernel<X><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(opNum, arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
    nd4j::DebugHelper::checkErrorCode(stream, "aggregateKernelGeneric(...) failed");
}

///////////////////////////////////////////////////////////////////////
template <typename X>
__device__ void AggregatedFunction<X>::aggregateBatch(int opNum, int numAggregates, 
                                                    int maxArgs, int maxShapes, 
                                                    int maxIntArrays, int maxIntArraySize, 
                                                    int maxIdx, int maxReals, 
                                                    void *ptrToArguments) {

    nd4j::PointersHelper<X> helper(ptrToArguments, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

    // TODO: we probably should lift this restriction
    __shared__ int *intArrays[32];

    __shared__ X **arguments;
    __shared__ Nd4jLong **shapes;
    __shared__ int *idxArg;
    __shared__ X *realArg;

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

        functions::aggregate::AggregatedFunction<X>::execCuda(opNum, arguments, helper.getNumArguments(r), shapes, helper.getNumShapeArguments(r), idxArg, helper.getNumIndexArguments(r), intArrays, helper.getNumIntArrayArguments(r), realArg, helper.getNumRealArguments(r));
    }
}

///////////////////////////////////////////////////////////////////////
template <typename X>
__global__ static void execAggregateBatch(int opNum, int numAggregates, 
                                        int maxArgs, int maxShapes, 
                                        int maxIntArrays, int maxIntArraySize, 
                                        int maxIdx, int maxReals, 
                                        void *ptrToArguments) {

    functions::aggregate::AggregatedFunction<X>::aggregateBatch(opNum, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments);
}

///////////////////////////////////////////////////////////////////////
template <typename X>
__host__ void AggregatedFunction<X>::aggregateBatchKernelGeneric(dim3& launchDims, cudaStream_t *stream, 
                                                    int opNum, int numAggregates, 
                                                    int maxArgs, int maxShapes, 
                                                    int maxIntArrays, int maxIntArraySize, 
                                                    int maxIdx, int maxReals, 
                                                    void *ptrToArguments) {

    execAggregateBatch<X><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(opNum, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments);
    nd4j::DebugHelper::checkErrorCode(stream, "aggregateBatchKernel(...) failed");
}





BUILD_SINGLE_TEMPLATE(template class AggregatedFunction, , FLOAT_TYPES);
}
}
