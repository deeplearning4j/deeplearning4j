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
// @author Yurii Shyrma, created on 15.11.2018
//

#include <loops/special_kernels.h>

namespace nd4j {

///////////////////////////////////////////////////////////////////////
    template<typename T>
    __device__ void concatKernelHStack(int numArrays,
                                       Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
                                       void *vz, Nd4jLong *zShapeInfo) {

        // we expect all data coming in as vectors, and z as 2D matrix
        // the only significant difference here is the fact that input lengths might be different
        auto z = reinterpret_cast<T *>(vz);
        auto inputShapes = (Nd4jLong **) inputShapeInfos;
        T **input = (T **) data;

        __shared__ int inputEWS;
        __shared__ int resultEWS;
        __shared__ int inputLength;

        if (threadIdx.x == 0) {
            resultEWS = shape::elementWiseStride(zShapeInfo);
        }
        __syncthreads();

        for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {

            __shared__ int baseIdx;
            if (threadIdx.x == 0) {
                baseIdx = 0;
                for (int f = 0; f < r; f++) {
                    baseIdx += shape::length(inputShapes[f]);
                }
            }
            __syncthreads();


            T *inputData = (T *) input[r];

            if (threadIdx.x == 0) {
                inputEWS = shape::elementWiseStride(inputShapes[r]);
                inputLength = shape::length(inputShapes[r]);
            }
            __syncthreads();

            for (int i = threadIdx.x; i < inputLength; i += blockDim.x) {
                z[baseIdx + i * resultEWS] = inputData[i * inputEWS];
            }
            __syncthreads();
        }
    }

///////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void execConcatKernelHStack(int numArrays,
                                           Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
                                           void *vz, Nd4jLong *zShapeInfo) {

        concatKernelHStack<T>(numArrays, data, inputShapeInfos, vz, zShapeInfo);
    }

///////////////////////////////////////////////////////////////////////
    template<typename T>
    __host__ void concatKernelHStackGeneric(dim3 &launchDims, cudaStream_t *stream,
                                            int numArrays,
                                            Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
                                            void *vz, Nd4jLong *zShapeInfo) {

        execConcatKernelHStack<T> << < launchDims.x, launchDims.y, launchDims.z, *stream >> >
                                                                                 (numArrays, data, inputShapeInfos, vz, zShapeInfo);
    }

    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT concatKernelHStackGeneric, (dim3 & launchDims, cudaStream_t * stream, int numArrays, Nd4jPointer * data, Nd4jPointer * inputShapeInfos, void * vz, Nd4jLong * zShapeInfo), LIBND4J_TYPES);
}