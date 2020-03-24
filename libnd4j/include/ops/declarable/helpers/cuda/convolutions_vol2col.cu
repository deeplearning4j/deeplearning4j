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
// vol [bS, iC, iD, iH, iW] is convoluted to col [bS, iC, kD, kH, kW, oD, oH, oW]
template <typename T>
static __global__ void vol2colCuda(const void* volume, const Nd4jLong* volShapeInfo, void* columns, const Nd4jLong* colShapeInfo,  const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const T* vol = reinterpret_cast<const T*>(volume);
          T* col = reinterpret_cast<T*>(columns);

    __shared__ int colRank, volRank;
    __shared__ Nd4jLong colLen, iD, iH, iW, *sharedMem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        volRank = 5;
        colRank = 8;

        colLen = shape::length(colShapeInfo);

        iD = volShapeInfo[3];
        iH = volShapeInfo[4];
        iW = volShapeInfo[5];
    }
    __syncthreads();

    const auto colInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(colInd >= colLen)
        return;

    auto coords = sharedMem + threadIdx.x * colRank;

    shape::index2coords(colInd, colShapeInfo, coords);

    // const auto colW = coords[7];
    // const auto colH = coords[6];
    // const auto colD = coords[5];
    // const auto kCol = coords[4];
    // const auto kRow = coords[3];
    // const auto kDep = coords[2];
    // const auto c    = coords[1];
    // const auto b    = coords[0];

    const auto colOffset = shape::getOffset(colShapeInfo, coords);

    coords[2] = -pD + coords[2] * dD + coords[5] * sD;     // const auto volDep = (-pD + kDep * dD) + colD * sD;
    coords[3] = -pH + coords[3] * dH + coords[6] * sH;     // const auto volRow = (-pH + kRow * dH) + colH * sH;
    coords[4] = -pW + coords[4] * dW + coords[7] * sW;     // const auto volCol = (-pW + kCol * dW) + colW * sW;

    if (static_cast<unsigned>(coords[2]) >= static_cast<unsigned>(iD) || static_cast<unsigned>(coords[3]) >= static_cast<unsigned>(iH) || static_cast<unsigned>(coords[4]) >= static_cast<unsigned>(iW))
        col[colOffset] = static_cast<T>(0.);
    else
        col[colOffset] = vol[shape::getOffset(volShapeInfo, coords)];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void vol2colCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const void* volume, const Nd4jLong* volShapeInfo,
                                      void* columns, const Nd4jLong* colShapeInfo,
                                const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    vol2colCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(volume, volShapeInfo, columns, colShapeInfo,  sD, sH, sW, pD, pH, pW, dD, dH, dW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::vol2col(sd::graph::Context& block, const NDArray& vol, NDArray& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    PointersManager manager(block.launchContext(), "vol2col");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (col.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = col.rankOf() * sizeof(Nd4jLong) * threadsPerBlock  + 128;

    NDArray::prepareSpecialUse({&col}, {&vol});
    BUILD_SINGLE_SELECTOR(vol.dataType(), vol2colCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), vol.getSpecialBuffer(), vol.getSpecialShapeInfo(), col.specialBuffer(), col.specialShapeInfo(), sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&col}, {&vol});

    manager.synchronize();
}

}
}