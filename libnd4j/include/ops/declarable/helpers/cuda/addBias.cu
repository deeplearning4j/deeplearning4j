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
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include<ops/declarable/helpers/addBias.h>
#include <PointersManager.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__global__ static void addBiasCuda( const void* vx, const Nd4jLong* xShapeInfo,
                                    const void* vy, const Nd4jLong* yShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo,
                                    const bool isNCHW) {

    // bias [oC]

    // if(input_rank == 4)
        // input and output have same shapes: [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
    // if(input_rank == 5)
        // input and output have same shapes: [bS, oD, oH, oW, oC] (NHWC) or [bS, oD, oC, oH, oW] (NCHW)

    const X* x = reinterpret_cast<const X*>(vx);
    const Y* y = reinterpret_cast<const Y*>(vy);
          X* z = reinterpret_cast<X*>(vz);

    __shared__ int rank, channelPosition, posOfNonUnityDim;
    __shared__ Nd4jLong *sharedMem, len;
    __shared__ bool xzSameOffsets, xzAreSame;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        rank = shape::rank(xShapeInfo);     // xRank == zRank
        xzSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
        len = shape::length(xShapeInfo);
        channelPosition = isNCHW ? 1 : rank - 1;        // second or last
        xzAreSame = x == z;

        shape::isCommonVector(yShapeInfo, posOfNonUnityDim);
    }
    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {

        shape::index2coords(i, xShapeInfo, coords);

        const auto xOffsets = shape::getOffset(xShapeInfo, coords);
        const auto zOffsets = xzSameOffsets ? xOffsets : shape::getOffset(zShapeInfo, coords);
        const auto yOffsets = coords[channelPosition] * shape::stride(yShapeInfo)[posOfNonUnityDim];

        if(xzAreSame)
            z[zOffsets] += static_cast<X>(y[yOffsets]);
        else
            z[zOffsets] = x[xOffsets] + static_cast<X>(y[yOffsets]);
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void addBiasCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                         const void* vx, const Nd4jLong* xShapeInfo,
                                         const void* vy, const Nd4jLong* yShapeInfo,
                                               void* vz, const Nd4jLong* zShapeInfo,
                                         const bool isNCHW) {

    addBiasCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, isNCHW);
}

//////////////////////////////////////////////////////////////////////////
void addBias(nd4j::graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW) {

    PointersManager manager(block.launchContext(), "addBias");

    const int threadsPerBlock = MAX_NUM_THREADS/2;
    const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = input.rankOf() * sizeof(Nd4jLong) * threadsPerBlock  + 128;

    NDArray::prepareSpecialUse({&output}, {&input, &bias});
    BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBiasCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), bias.getSpecialBuffer(), bias.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), isNCHW), FLOAT_TYPES, FLOAT_TYPES);
    NDArray::registerSpecialUse({&output}, {&input, &bias});

    manager.synchronize();
}

}
}
}