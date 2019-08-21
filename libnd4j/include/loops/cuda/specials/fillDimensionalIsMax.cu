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


////////////////////////////////////////////////////////////////////////
    template <typename T>
    __device__ void fillDimensionalIsMax(void *vdX,
                                         void *vdZ, Nd4jLong *zShapeInfo,
                                         Nd4jLong *tadOnlyShapeInfo,
                                         int *dimension, int dimensionLength,
                                         Nd4jLong *tadOffsets) {

        auto dX = reinterpret_cast<Nd4jLong *>(vdX);
        auto dZ = reinterpret_cast<T *>(vdZ);

        __shared__ int tadLength;
        __shared__ int tadEWS;
        __shared__ int numTads;

        if (threadIdx.x == 0) {
            tadLength = shape::length(tadOnlyShapeInfo);//shape::tadLength(zShapeInfo, dimension, dimensionLength);
            tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
            numTads = shape::length(zShapeInfo) / tadLength;
        }
        __syncthreads();

        for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
            auto tadOffsetForBlock = tadOffsets[r];
            auto highestElement = dX[r];

            if (dimensionLength > 1 || tadEWS < 1) {

                for (Nd4jLong e = threadIdx.x; e < tadLength; e += blockDim.x) {
                    auto xOffset = tadOffsetForBlock + shape::getIndexOffset(e, tadOnlyShapeInfo, tadLength);
                    dZ[xOffset] = (e == highestElement ? (T) 1 : (T) 0);
                }
            } else {
                for (Nd4jLong e = threadIdx.x; e < tadLength; e += blockDim.x) {
                    // so, we just set dZ[e] for each TAD. Sure, e should be replaced with
                    auto idx = tadOffsetForBlock + (e * tadEWS);
                    dZ[idx] = (e == highestElement ? (T) 1 : (T) 0);
                }
            }
        }
    }


////////////////////////////////////////////////////////////////////////
    template <typename T>
    __global__ void execfillDimensionalIsMax(void *dX,
                                             void *dZ, Nd4jLong *zShapeInfo,
                                             Nd4jLong *tadOnlyShapeInfo,
                                             int *dimension, int dimensionLength,
                                             Nd4jLong *tadOffsets) {

        fillDimensionalIsMax<T>(dX, dZ, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    __host__ void fillDimensionalIsMaxGeneric(dim3 &launchDims, cudaStream_t *stream,
                                              void *dX,
                                              void *dZ, Nd4jLong *zShapeInfo,
                                              Nd4jLong *tadOnlyShapeInfo,
                                              int *dimension, int dimensionLength,
                                              Nd4jLong *tadOffsets) {

        execfillDimensionalIsMax<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(dX, dZ, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
        nd4j::DebugHelper::checkErrorCode(stream, "fillDimensionalIsMax(...) failed");
    }
    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT fillDimensionalIsMaxGeneric, (dim3& launchDims, cudaStream_t *stream, void *dX, void *dZ, Nd4jLong *zShapeInfo, Nd4jLong *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOffsets), LIBND4J_TYPES);
}