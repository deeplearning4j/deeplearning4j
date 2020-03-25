/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

#include <ops/declarable/helpers/convolutions.h>
#include <helpers/PointersManager.h>

namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void upsampling2dBPCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const bool isNCHW) {

    // x (gradO) has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    // z (gradI) has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank, dimIH;
    __shared__ uint factorH, factorW;
    __shared__ Nd4jLong zLen, *sharedMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        dimIH = isNCHW ? 2 : 1;
        zLen  = shape::length(zShapeInfo);
        rank  = 4;

        factorH = xShapeInfo[dimIH + 1] / zShapeInfo[dimIH + 1];
        factorW = xShapeInfo[dimIH + 2] / zShapeInfo[dimIH + 2];
    }
    __syncthreads();

    const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(zInd >= zLen)
        return;

    auto coords = sharedMem + threadIdx.x * rank;

    shape::index2coords(zInd, zShapeInfo, coords);

    const auto zOffset = shape::getOffset(zShapeInfo, coords);

    z[zOffset] = 0;

    const Nd4jLong zCoord2 = coords[dimIH]     * factorH;
    const Nd4jLong zCoord3 = coords[dimIH + 1] * factorW;

    for(coords[dimIH] = zCoord2; coords[dimIH] < zCoord2 + factorH; ++coords[dimIH])
        for(coords[dimIH + 1] = zCoord3; coords[dimIH + 1] < zCoord3 + factorW; ++coords[dimIH + 1])
            z[zOffset] += x[shape::getOffset(xShapeInfo, coords)];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling2dBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                       const void* vx, const Nd4jLong* xShapeInfo,
                                             void* vz, const Nd4jLong* zShapeInfo,
                                       const bool isNCHW) {

    upsampling2dBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, isNCHW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::upsampling2dBP(sd::graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {

    PointersManager manager(block.launchContext(), "upsampling2d_bp");

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (gradI.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = gradI.rankOf() * sizeof(Nd4jLong) * threadsPerBlock + 128;

    NDArray::prepareSpecialUse({&gradI}, {&gradO});
    BUILD_SINGLE_SELECTOR(gradI.dataType(), upsampling2dBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), gradO.getSpecialBuffer(), gradO.getSpecialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), isNCHW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&gradI}, {&gradO});

    manager.synchronize();
}

}
}