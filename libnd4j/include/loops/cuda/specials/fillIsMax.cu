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
    template <typename T>
    __global__ void execFillIsMax(void *vdZ, const Nd4jLong *xShapeInfo, Nd4jLong length, long idx) {
        auto dz = reinterpret_cast<T*>(vdZ);
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        for (Nd4jLong i = tid; i < length; i += blockDim.x * gridDim.x)
            dz[shape::getIndexOffset(i, xShapeInfo)] = (i == idx ? (T) 1 : (T) 0);
    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    __host__ void fillIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, const Nd4jLong *xShapeInfo, Nd4jLong length, long idx) {
        execFillIsMax<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(dx, xShapeInfo, length, idx);
        sd::DebugHelper::checkErrorCode(stream, "fillIsMax(...) failed");
    }


    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT fillIsMaxGeneric, (dim3& launchDims, cudaStream_t *stream, void* dz, const Nd4jLong *zShapeInfo, Nd4jLong length, long idx), LIBND4J_TYPES);
}