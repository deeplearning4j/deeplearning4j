/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

namespace sd {

////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void execConvertToHalf(void *dx, Nd4jLong n, half *dz) {
        auto x = reinterpret_cast<T *>(dx);
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        for (Nd4jLong i = tid; i < n; i += blockDim.x * gridDim.x)
            dz[i] = __float2half(static_cast<T>(x[i]));
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    __host__ void convertToHalfGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, Nd4jLong n, half *dz) {
        execConvertToHalf<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(dx, n, dz);
        sd::DebugHelper::checkErrorCode(stream, "convertToHalfs(...) failed");
    }

    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT convertToHalfGeneric, (dim3 & launchDims, cudaStream_t * stream, void * dx, Nd4jLong n, half * dz), LIBND4J_TYPES);

}