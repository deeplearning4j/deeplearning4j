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
    __device__ void concatKernelScalar(int numArrays, Nd4jPointer *data, void *vz) {

        auto z = static_cast<T *>(vz);
        Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto input = reinterpret_cast<T **>(data);

        for (int i = tid; i < numArrays; i += blockDim.x * gridDim.x)
            z[i] = input[i][0];
    }

///////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void execConcatKernelScalar(int numArrays, Nd4jPointer *data, void *vz) {

        concatKernelScalar<T>(numArrays, data, vz);
    }

///////////////////////////////////////////////////////////////////////
    template<typename T>
    __host__ void
    concatKernelScalarGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Nd4jPointer *data, void *vz) {

        execConcatKernelScalar<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(numArrays, data, vz);
        nd4j::DebugHelper::checkErrorCode(stream, "concatScalar(...) failed");
    }

    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT concatKernelScalarGeneric, (dim3 & launchDims, cudaStream_t * stream, int numArrays, Nd4jPointer * data, void * vz), LIBND4J_TYPES);
}