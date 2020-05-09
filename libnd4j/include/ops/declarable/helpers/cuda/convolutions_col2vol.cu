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
#include <math/templatemath.h>

namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// columns [bS, iC, kD, kH, kW, oD, oH, oW] to be de-convoluted to volume [bS, iC, iD, iH, iW]
template <typename T>
static __global__ void col2volCuda(const void* columns, const Nd4jLong* colShapeInfo, void* volume, const Nd4jLong* volShapeInfo,  const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const T* col = reinterpret_cast<const T*>(columns);
          T* vol = reinterpret_cast<T*>(volume);

    __shared__ uint kD, kH, kW, oD, oH, oW, *sharedMem;
    __shared__ Nd4jLong volLen;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<uint*>(shmem);

        oD = colShapeInfo[6];
        oH = colShapeInfo[7];
        oW = colShapeInfo[8];

        kD = dD * (colShapeInfo[3] - 1) + 1;
        kH = dH * (colShapeInfo[4] - 1) + 1;
        kW = dW * (colShapeInfo[5] - 1) + 1;

        volLen  = shape::length(volShapeInfo);
    }
    __syncthreads();

    auto coords = sharedMem + threadIdx.x * 8;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < volLen; i += gridDim.x * blockDim.x) {

        shape::index2coords(i, volShapeInfo, coords);

        const auto volOffset = shape::getOffset(volShapeInfo, coords);

        const auto bSiCoffset = coords[0] * colShapeInfo[9] + coords[1] * colShapeInfo[10];

        const uint imD = coords[2] + pD;
        const uint imH = coords[3] + pH;
        const uint imW = coords[4] + pW;

        const uint colDstart = (imD < kD) ? 0 : (imD - kD) / sD + 1;
        const uint colHstart = (imH < kH) ? 0 : (imH - kH) / sH + 1;
        const uint colWstart = (imW < kW) ? 0 : (imW - kW) / sW + 1;

        const uint colDend = sd::math::nd4j_min<uint>(imD / sD + 1, oD);
        const uint colHend = sd::math::nd4j_min<uint>(imH / sH + 1, oH);
        const uint colWend = sd::math::nd4j_min<uint>(imW / sW + 1, oW);

        T val = 0;

        for(uint colD = colDstart; colD < colDend; ++colD) {
            coords[2] = imD - colD * sD;
            if(coords[2] % dD != 0) continue;

            for(uint colH = colHstart; colH < colHend; ++colH) {
                coords[3] = imH - colH * sH;
                if(coords[3] % dH != 0) continue;

                for(uint colW = colWstart; colW < colWend; ++colW) {
                    coords[4] = imW - colW * sW;
                    if(coords[4] % dW != 0) continue;

                    val += col[bSiCoffset + (coords[2]/dD)*colShapeInfo[11] + (coords[3]/dH)*colShapeInfo[12] + (coords[4]/dW)*colShapeInfo[13] + colD*colShapeInfo[14] + colH*colShapeInfo[15] + colW*colShapeInfo[16]];

                }
            }
        }

        vol[volOffset] = val;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void col2volCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const void* columns, const Nd4jLong* colShapeInfo,
                                      void* volume, const Nd4jLong* volShapeInfo,
                                const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    col2volCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(columns, colShapeInfo, volume, volShapeInfo, sD, sH, sW, pD, pH, pW, dD, dH, dW);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::col2vol(sd::graph::Context& block, const NDArray& col, NDArray& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    PointersManager manager(block.launchContext(), "col2vol");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (vol.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = col.rankOf() * sizeof(uint) * threadsPerBlock  + 256;

    NDArray::prepareSpecialUse({&vol}, {&col});
    BUILD_SINGLE_SELECTOR(vol.dataType(), col2volCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), col.specialBuffer(), col.specialShapeInfo(), vol.specialBuffer(), vol.specialShapeInfo(), sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&vol}, {&col});

    manager.synchronize();
}

}
}
