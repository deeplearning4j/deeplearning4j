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
//  @author raver119@gmail.com
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/axis.h>
#include <helpers/PointersManager.h>
#include <helpers/TAD.h>
#include <array>
#include <helpers/ConstantTadHelper.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static __global__ void globalExtractPatches_(void *vinput, Nd4jLong *xTadShape, Nd4jLong *xTadOffsets, void *voutput, Nd4jLong *zTadShape, Nd4jLong *zTadOffsets, const int numTads, const int sizeRow, const int sizeCol, const int stradeRow, const int stradeCol, const int rateRow, const int rateCol, const bool theSame, const int lastDim, const int rowDim, const int colDim) {
        auto input = reinterpret_cast<T*>(vinput);
        auto output = reinterpret_cast<T*>(voutput);

        const int warpSize = lastDim;
        const int tid = blockIdx.x * gridDim.x + threadIdx.x;
        const int warpIdx = tid / warpSize;
        const int warpPos = tid % warpSize;
        const int numWarps = (gridDim.x * blockDim.x) / warpSize;
        const int patchLength = shape::length(zTadShape);

        auto xShape = shape::shapeOf(xTadShape);
        auto xStride = shape::stride(xTadShape);
        auto xRank = shape::rank(xTadShape);

        for (int e = warpIdx; e < numTads; e += numWarps) {
            auto patch = input + xTadOffsets[e];
            auto matrix = output + zTadOffsets[e];
                int iter = 0;

                for (int i = 0; i < rowDim; i += stradeRow)
                    for (int j = 0; j < colDim; j += stradeCol)
                        for (int l = 0; l < sizeRow && l + i < rowDim; l++)
                            for (int m = 0; m < sizeCol && m + j < colDim; m++) {
                                auto pos = warpPos + (iter * lastDim);

                                if (pos < patchLength) {
                                    auto x = i + rateRow * l;
                                    auto y = j + m * rateCol;
                                    Nd4jLong xIndex[3] = {x, y, warpPos};

                                    matrix[shape::getIndexOffset(pos, zTadShape, patchLength)] = patch[shape::getOffset(0, xShape, xStride, xIndex, xRank)];
                                } else {
                                    // early loop termination
                                    i = rowDim;
                                    j = colDim;
                                    l = sizeRow;
                                    m = sizeCol;
                                }

                                iter++;
                            }

            __syncthreads();
        }
    }

    template <typename T>
    static void _extractPatches(nd4j::LaunchContext * context, NDArray* images, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
        std::array<int, 3> restDims = {1, 2, 3};

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(images->getShapeInfo(), restDims.data(), restDims.size());
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), restDims.data(), restDims.size());


        PointersManager manager(context, "helpers::extractPatches");

        int lastDim = images->sizeAt(3);
        int rowDim = images->sizeAt(1);
        int colDim = images->sizeAt(2);

        globalExtractPatches_<T><<<512, 512, 1024, *context->getCudaStream()>>>(images->getSpecialBuffer(), packX.specialShapeInfo(), packX.specialOffsets(), output->getSpecialBuffer(), packZ.specialShapeInfo(), packZ.specialOffsets(), packX.numberOfTads(), sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame, lastDim, rowDim, colDim);

        output->tickWriteDevice();

        manager.synchronize();
    }
    BUILD_SINGLE_TEMPLATE(template void _extractPatches, (nd4j::LaunchContext * context, NDArray* input, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame), LIBND4J_TYPES);



    void extractPatches(nd4j::LaunchContext * context, NDArray* images, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
        auto xType = images->dataType();

        BUILD_SINGLE_SELECTOR(xType, _extractPatches, (context, images, output, sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame), LIBND4J_TYPES);
    }
}
}
}