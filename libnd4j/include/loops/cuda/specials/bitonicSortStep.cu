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
// @author Yurii Shyrma, created on 28.11.2018
//

#include <ops/specials_cuda.h>

//////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ void bitonicSortStepKernel(void *vx, Nd4jLong *xShapeInfo, int j, int k, int length, bool descending) {

    auto x = static_cast<T*>(vx);

    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ Nd4jLong xLength;
    if (threadIdx.x == 0)
        xLength = shape::length(xShapeInfo);

    __syncthreads();


    if (i >= length)
        return;

    ixj = i^j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj)>i) {
        int posI = getDevicePosition(xShapeInfo, i, xLength);
        int posIXJ = getDevicePosition(xShapeInfo, ixj, xLength);

        if ((i&k)==0) {
            /* Sort ascending */
            if (!descending == (x[posI]>x[posIXJ])) {
                /* exchange(i,ixj); */
                T temp = x[posI];
                x[posI] = x[posIXJ];
                x[posIXJ] = temp;
            }
        } else if ((i&k)!=0) {
            /* Sort descending */
            if (!descending == (x[posI]<x[posIXJ])) {
                /* exchange(i,ixj); */
                T temp = x[posI];
                x[posI] = x[posIXJ];
                x[posIXJ] = temp;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void execBitonicSortStepKernel(void *vx, Nd4jLong *xShapeInfo, int j, int k, int length, bool descending) {

    bitonicSortStepKernel<T>(vx, xShapeInfo, j, k, length, descending);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__host__ void bitonicSortStepGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, int j, int k, int length, bool descending) {

    execBitonicSortStepKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, j, k, length, descending);
    nd4j::DebugHelper::checkErrorCode(stream, "bitonicSortStep(...) failed");
}
BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT bitonicSortStepGeneric, (dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, int j, int k, int length, bool descending), LIBND4J_TYPES);
