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
// @author Yurii Shyrma, created on 28.11.2018
//

#include <ops/specials_cuda.h>

//////////////////////////////////////////////////////////////////////////
template<typename T>
__device__
void oesTadKernel(void *vx, Nd4jLong *xShapeInfo, 
                int *dimension, int dimensionLength, 
                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, 
                bool descending) {

    auto x = static_cast<T*>(vx);
    
    __shared__ int xLength;
    __shared__ int xTadLength;
    __shared__ int numTads;
    __shared__ T *shmem;
    if (threadIdx.x == 0) {
        xLength = shape::length(xShapeInfo);
        xTadLength = shape::length(tadShapeInfo);
        numTads = xLength / xTadLength;

        extern __shared__ unsigned char shrd[];
        shmem = (T *) shrd;
    }
    __syncthreads();

    int limit = nd4j::math::nd4j_max<int>((int) xTadLength, blockDim.x);

    if (limit > blockDim.x)
        limit = limit + (blockDim.x - (xTadLength % blockDim.x));

    //if (threadIdx.x == 0 && blockIdx.x == 0)
    //    printf("Tad length: %i; blockDim.x: %i; limit: %i;\n", (int) xTadLength, blockDim.x, limit);

    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
        auto dx = x + tadOffsets[r];

        // this is general loop, we go uncached
        int iterations = xTadLength - 2;

        int rem = xTadLength % 2;

        for (int i = 0; i < iterations; i++) {
            if (i % 2 == 0) {
                for (int tid = threadIdx.x; tid < limit; tid += blockDim.x) {
                    auto top = 2 * tid + 1;
                    if (top < xTadLength) {
                        auto t0 = getDevicePosition(tadShapeInfo, top - 1, xTadLength);
                        auto t1 = getDevicePosition(tadShapeInfo, top, xTadLength);

                        //if (r == 0 && i == 0 && tid > 375)
                        //    printf("LTID: [%i]; t0: [%i]; t1: [%i];\n", tid, (int) t0, (int) t1);

                        if (!descending == (dx[t0] > dx[t1])) {
                            T dt0 = dx[t0];
                            dx[t0] = dx[t1];
                            dx[t1] = dt0;
                        }
                    }
                }
            } else {
                for (int tid = threadIdx.x; tid < limit; tid += blockDim.x) {
                    auto top = 2 * tid + 2;
                    if (top < xTadLength) {
                        auto t0 = getDevicePosition(tadShapeInfo, top - 1, xTadLength);
                        auto t1 = getDevicePosition(tadShapeInfo, top, xTadLength);

                        //if (r == 0 && i == 1 && tid > 375)
                        //    printf("RTID: [%i]; t0: [%i]; t1: [%i];\n", tid, (int) t0, (int) t1);

                        if (!descending == (dx[t0] > dx[t1])) {
                            T dt0 = dx[t0];
                            dx[t0] = dx[t1];
                            dx[t1] = dt0;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void execOesTadKernel(void *vx, Nd4jLong *xShapeInfo, 
                                int *dimension, int dimensionLength, 
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, 
                                bool descending) {

    oesTadKernel<T>(vx, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__host__ void oesTadGeneric(dim3 &launchDims, cudaStream_t *stream,
                                void *vx, Nd4jLong *xShapeInfo, 
                                int *dimension, int dimensionLength, 
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, 
                                bool descending) {

    execOesTadKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}
BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT oesTadGeneric, (dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending), LIBND4J_TYPES);
