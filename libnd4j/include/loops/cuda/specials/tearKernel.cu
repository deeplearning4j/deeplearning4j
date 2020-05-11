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

namespace sd {

////////////////////////////////////////////////////////////////////////
    template<typename T>
    __device__ void
    tearKernel(void *vx, Nd4jLong const* xShapeInfo, Nd4jPointer *targets, Nd4jLong const* zShapeInfo, Nd4jLong const* tadShapeInfo,
               Nd4jLong const* tadOffsets) {



        __shared__         Nd4jLong tadLength;
        __shared__ int tadEWS;
        __shared__ int zEWS;
//        __shared__ int tadRank;
        __shared__         Nd4jLong numTads;
//        __shared__ int zRank;
//        __shared__        Nd4jLong *tadShape;
//        __shared__        Nd4jLong *tadStride;
//        __shared__        Nd4jLong const* zShape;
//        __shared__        Nd4jLong const* zStride;
        __shared__ T* x;
        if (threadIdx.x == 0) {
            tadLength = shape::length(tadShapeInfo);
            tadEWS = shape::elementWiseStride(tadShapeInfo);
            zEWS = shape::elementWiseStride(zShapeInfo);
            numTads = shape::length(xShapeInfo) / tadLength;
            x = static_cast<T *>(vx);
        }
        __syncthreads();

        for (Nd4jLong r = blockIdx.x; r < numTads; r += gridDim.x) {
            T *z = (T *) targets[r];
            T *s = x + tadOffsets[r];

            if (zEWS > 0 && tadEWS > 0) {
                for (Nd4jLong i = threadIdx.x; i < tadLength; i += blockDim.x)
                    z[i * zEWS] = s[i * tadEWS];
            } else {

                for (Nd4jLong j = threadIdx.x; j < tadLength; j += blockDim.x) {
                    auto xOffset = shape::getIndexOffset(j, tadShapeInfo);
                    auto zOffset = shape::getIndexOffset(j, zShapeInfo);

                    z[zOffset] = s[xOffset];
                }
            }
        }
    }


////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void
    execTearKernel(void *vx, Nd4jLong const* xShapeInfo, Nd4jPointer *targets, Nd4jLong const* zShapeInfo, Nd4jLong const* tadShapeInfo,
                   Nd4jLong const* tadOffsets) {

        tearKernel<T>(vx, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    __host__ void tearKernelGeneric(dim3 &launchDims, cudaStream_t *stream,
                                    void *vx, Nd4jLong const* xShapeInfo,
                                    Nd4jPointer *targets, Nd4jLong const* zShapeInfo,
                                    Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets) {

        execTearKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
        sd::DebugHelper::checkErrorCode(stream, "tear(...) failed");
    }

    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT tearKernelGeneric, (dim3 & launchDims, cudaStream_t * stream, void * vx, Nd4jLong const* xShapeInfo, Nd4jPointer *targets, Nd4jLong const* zShapeInfo, Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets), LIBND4J_TYPES);
}