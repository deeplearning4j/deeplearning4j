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
// @author Yurii Shyrma, created on 15.11.2018
//

#include <loops/special_kernels.h>

namespace nd4j {

///////////////////////////////////////////////////////////////////////
    template<typename T>
    __device__ void concatKernelVStack(int numArrays,
                                       Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
                                       void *vz, Nd4jLong *zShapeInfo) {

        /*
         this is special case for concat: we group bunch of vectors into 2D matrix
         also: we expect each inputShapeInfo to have EWS, be a vector, and have equal size
         */
        auto z = static_cast<T *>(vz);

        auto inputShapes = (Nd4jLong **) inputShapeInfos;
        T **input = (T **) data;

        __shared__ int inputEWS;
        __shared__ int resultEWS;
        __shared__ int inputLength;

        if (threadIdx.x == 0) {
            inputLength = shape::length(inputShapes[0]);
            inputEWS = shape::elementWiseStride(inputShapes[0]);
            resultEWS = shape::elementWiseStride(zShapeInfo);
        }
        __syncthreads();

        for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {

            int zOffset = r * inputLength * resultEWS;
            T *inputData = (T *) input[r];

            for (int i = threadIdx.x; i < inputLength; i += blockDim.x) {
                z[zOffset + i * resultEWS] = inputData[i * inputEWS];
            }
        }
    }

///////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void execConcatKernelVStack(int numArrays,
                                           Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
                                           void *vz, Nd4jLong *zShapeInfo) {

        concatKernelVStack<T>(numArrays, data, inputShapeInfos, vz, zShapeInfo);
    }


///////////////////////////////////////////////////////////////////////
    template<typename T>
    __host__ void concatKernelVStackGeneric(dim3 &launchDims, cudaStream_t *stream,
                                            int numArrays,
                                            Nd4jPointer *data, Nd4jPointer *inputShapeInfos,
                                            void *vz, Nd4jLong *zShapeInfo) {

        execConcatKernelVStack<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(numArrays, data, inputShapeInfos, vz, zShapeInfo);
        nd4j::DebugHelper::checkErrorCode(stream, "concatVStack(...) failed");
    }

    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT concatKernelVStackGeneric, (dim3 & launchDims, cudaStream_t * stream, int numArrays, Nd4jPointer * data, Nd4jPointer * inputShapeInfos, void * vz, Nd4jLong *zShapeInfo), LIBND4J_TYPES);
}