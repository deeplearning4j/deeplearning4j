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
    T *dx;
    if (threadIdx.x == 0) {
        xLength = shape::length(xShapeInfo);
        xTadLength = shape::length(tadShapeInfo);
        numTads = xLength / xTadLength;

        extern __shared__ unsigned char shrd[];
        shmem = (T *) shrd;
    }
    __syncthreads();

    T dt0, dt1;

    int limit = nd4j::math::nd4j_max<int>((int) xTadLength, blockDim.x);

    if (limit > blockDim.x)
        limit = limit + (blockDim.x - (xTadLength % blockDim.x));

    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
        dx = x + tadOffsets[r];

        // this is general loop, we go uncached
        int rem = xTadLength % 2;

        for (int i = 0; i < (xTadLength / 2) + rem; i++) {
            // since we can have TAD larger then blockDim, we'll have this loop here
            for (int tid = threadIdx.x; tid < limit; tid += blockDim.x ) {
                if((!(tid & 1)) && tid < xTadLength - 1) {
                    int t0 = getDevicePosition(tadShapeInfo, tid, xTadLength);
                    int t1 = getDevicePosition(tadShapeInfo, tid+1, xTadLength);

                    dt0 = dx[t0];
                    dt1 = dx[t1];

                    if(!descending == (dt0 > dt1)) {
                        dx[t1] = dt0;
                        dx[t0] = dt1;
                    }
                }

                __syncthreads();
            }

            for (int tid = threadIdx.x; tid < limit; tid += blockDim.x ) {
                if((tid & 1) && tid < xTadLength - 1) {
                    int t0 = getDevicePosition(tadShapeInfo, tid, xTadLength);
                    int t1 = getDevicePosition(tadShapeInfo, tid+1, xTadLength);

                    dt0 = dx[t0];
                    dt1 = dx[t1];

                    if(!descending == (dt0 > dt1)) {
                        dx[t1] = dt0;
                        dx[t0] = dt1;
                    }
                }
                __syncthreads();
            }
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
