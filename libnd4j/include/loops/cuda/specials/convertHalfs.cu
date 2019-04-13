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
    __device__ void convertHalfs(half *dx, Nd4jLong n, void *dz) {

        auto z = reinterpret_cast<T *>(dz);
        int tid = threadIdx.x + blockIdx.x * gridDim.x;

        for (Nd4jLong i = tid; i < n; i += blockDim.x * gridDim.x)
            z[i] = static_cast<T>(__half2float(dx[i]));
    }

///////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void execConvertHalfs(half *dx, Nd4jLong n, void *dz) {

        convertHalfs<T>(dx, n, dz);
    }


///////////////////////////////////////////////////////////////////////
    template<typename T>
    __host__ void convertHalfsToGeneric(dim3 &launchDims, cudaStream_t *stream, half *dx, Nd4jLong n, void *dz) {

        execConvertHalfs<T> << < launchDims.x, launchDims.y, launchDims.z, *stream >> > (dx, n, dz);
    }

    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT convertHalfsToGeneric, (dim3 & launchDims, cudaStream_t * stream, half * dx, Nd4jLong n, void * dz), LIBND4J_TYPES);
}