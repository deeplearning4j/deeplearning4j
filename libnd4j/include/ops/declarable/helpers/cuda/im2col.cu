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
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/im2col.h>
#include <PointersManager.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// input [bS, iC, iH, iW] is convoluted to output [bS, iC, kH, kW, oH, oW]
template <typename T>
__global__ static void im2colCuda(const void *image, void *columns,
                                  const Nd4jLong *imShapeInfo, const Nd4jLong *colShapeInfo,
                                  const int sH, const int sW,
                                  const int pH, const int pW,
                                  const int dH, const int dW,
                                  const double zeroPadValD) {

    T zeroPadVal = static_cast<T>(zeroPadValD); //Value to use when value is padding. Usually 0 but not always
    const auto im  = reinterpret_cast<const T*>(image);
          auto col = reinterpret_cast<T*>(columns);

    __shared__ Nd4jLong colLen, *sharedMem, iH, iW;
    __shared__ int imRank, colRank;
    __shared__ char colOrder;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        colRank = 6;
        imRank  = 4;

        colLen   = shape::length(colShapeInfo);
        colOrder = shape::order(colShapeInfo);

        iH = imShapeInfo[3];
        iW = imShapeInfo[4];
    }

    __syncthreads();

    const auto colInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(colInd >= colLen)
        return;

    auto coords = sharedMem + threadIdx.x * colRank;

    shape::index2coords(colRank, colShapeInfo + 1, colInd, colLen, coords, colOrder);

    const auto colOffset = shape::getOffset(0, colShapeInfo + 1, colShapeInfo + colRank + 1, coords, colRank);

    coords[2] = (-pH + coords[2] * dH) + coords[4] * sH;   // imH
    coords[3] = (-pW + coords[3] * dW) + coords[5] * sW;   // imW

    if (static_cast<unsigned>(coords[2]) >= static_cast<unsigned>(iH) || static_cast<unsigned>(coords[3]) >= static_cast<unsigned>(iW))
        col[colOffset] = zeroPadVal;
    else
        col[colOffset] = im[shape::getOffset(0, imShapeInfo + 1, imShapeInfo + imRank + 1, coords, imRank)];
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void im2colCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, nd4j::LaunchContext & context, const void *image, void *columns, const Nd4jLong *imShapeInfo, const Nd4jLong *colShapeInfo, int sH, int sW, int pH, int pW, int dH, int dW, double zeroPadVal) {
    im2colCuda<T><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Nd4jLong) * 6 /* rank of columns = 6 */, *context.getCudaStream()>>>(image, columns, imShapeInfo, colShapeInfo, sH, sW, pH, pW, dH, dW, zeroPadVal);
}

//////////////////////////////////////////////////////////////////////////
void im2col(nd4j::LaunchContext& context, const NDArray& image, NDArray& columns, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const NDArray& arrZeroPadVal) {

    PointersManager manager(&context, "im2col");

    const int threadsPerBlock = 512;
    const int blocksPerGrid = (columns.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    NDArray::prepareSpecialUse({&columns}, {&image});
    BUILD_SINGLE_SELECTOR(columns.dataType(), im2colCudaLauncher, (blocksPerGrid, threadsPerBlock, context, image.getSpecialBuffer(), columns.getSpecialBuffer(), image.getSpecialShapeInfo(), columns.getSpecialShapeInfo(), sH, sW, pH, pW, dH, dW, arrZeroPadVal.e<double>(0)), FLOAT_TYPES);
    NDArray::registerSpecialUse({&columns}, {&image});

    manager.synchronize();
}





}
}
}