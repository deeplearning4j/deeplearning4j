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
    __shared__ T *dx;
    if (threadIdx.x == 0) {
        xLength = shape::length(xShapeInfo);
        xTadLength = shape::length(tadShapeInfo);//shape::tadLength(xShapeInfo, dimension, dimensionLength);
        numTads = xLength / xTadLength;

        extern __shared__ unsigned char shrd[];
        shmem = (T *) shrd;
    }
    __syncthreads();

    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
        if (threadIdx.x == 0) {
            dx = x + tadOffsets[r];
        }
        __syncthreads();

        // this is general loop, we go uncached

        int rem = xTadLength % 2;
        if (xTadLength > 1024) {

            for (int i = 0; i < (xTadLength / 2) + rem; i++) {

                // since we can have TAD larger then blockDim, we'll have this loop here
                for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x ) {
                    if((!(tid & 1)) && tid < xTadLength - 1) {
                        int t0 = getDevicePosition(tadShapeInfo, tid);
                        int t1 = getDevicePosition(tadShapeInfo, tid+1);

                        if(!descending == (dx[t0] > dx[t1])) {
                            T temp = dx[t1];
                            dx[t1] = dx[t0];
                            dx[t0] = temp;
                        }
                    }
                }
                __syncthreads();

                for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x ) {
                    if((tid & 1) && tid < xTadLength - 1) {
                        int t0 = getDevicePosition(tadShapeInfo, tid);
                        int t1 = getDevicePosition(tadShapeInfo, tid+1);

                        if(!descending == (dx[t0] > dx[t1])) {
                            T temp = dx[t1];
                            dx[t1] = dx[t0];
                            dx[t0] = temp;
                        }
                    }
                }
                __syncthreads();
            }

        } else {
            // we just load up to 1024 elements into shared memory, and sort will be applied there
            for (int e = threadIdx.x; e < xTadLength; e += blockDim.x)
                shmem[e] = dx[getDevicePosition(tadShapeInfo, e)];

            __syncthreads();


            for(int i=0; i < (xTadLength / 2) + rem; i++) {

                for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x ) {
                    if((!(tid & 1)) && tid < xTadLength - 1) {
                        int t0 = tid;
                        int t1 = tid+1;

                        if(!descending == (shmem[t0] > shmem[t1])) {
                            T temp = shmem[t1];
                            shmem[t1] = shmem[t0];
                            shmem[t0] = temp;
                        }
                    }
                    __syncthreads();

                    if((tid & 1) && tid < xTadLength - 1) {
                        int t0 = tid;
                        int t1 = tid+1;

                        if(!descending == (shmem[t0] > shmem[t1])) {
                            T temp = shmem[t1];
                            shmem[t1] = shmem[t0];
                            shmem[t0] = temp;
                        }
                    }
                    __syncthreads();
                }
            }

            // we're dumping our shared memory back to device memory
            for (int e = threadIdx.x; e < xTadLength; e += blockDim.x)
                dx[getDevicePosition(tadShapeInfo, e)] = shmem[e];

            __syncthreads();
        }

        __syncthreads();
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
