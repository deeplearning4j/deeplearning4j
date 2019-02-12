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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/axis.h>
#include <helpers/PointersManager.h>
#include <helpers/TAD.h>
#include <array>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static __global__ void globalExtractPatches_(void *vinput, Nd4jLong *xTadShape, Nd4jLong *xTadOffsets, void *voutput, Nd4jLong *zTadShape, Nd4jLong *zTadOffsets, const int numTads, const int sizeRow, const int sizeCol, const int stradeRow, const int stradeCol, const int rateRow, const int rateCol, const bool theSame, const int lastDim, const int rowDim, const int colDim) {
        auto input = reinterpret_cast<T*>(vinput);
        auto output = reinterpret_cast<T*>(voutput);

        const int warpSize = 32;
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
            int startRow = 0;
            int startCol = 0;
            int pos = 0;

            for (int i = warpPos; i < patchLength; i += warpSize) {
                Nd4jLong xIndex[3] = {0, 0, 0};
                matrix[shape::getIndexOffset(i, zTadShape, patchLength)] = patch[shape::getOffset(0, xShape, xStride, xIndex, xRank)];
            }

            /*
            for (int i = 0; i < rowDim; i += stradeRow)
                for (int j = 0; j < colDim; j += stradeCol)
                    for (int l = 0; l < sizeRow && l + i < rowDim; l++)
                        for (int m = 0; m < sizeCol && m + j < colDim; m++) {
                            for (int k = 0; k < lastDim; ++k) {
                                outMatrix->p<T>(pos++, patch->e<T>(i + rateRow * l, j + m * rateCol, k));
                                if (pos >= outMatrix->lengthOf()) { k = lastDim; m = sizeCol; l = sizeRow; j = colDim; i = rowDim; }
                            }
                        }
            */

            __syncthreads();
        }
    }

    template <typename T>
    static void _extractPatches(graph::LaunchContext* context, NDArray* images, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
        std::array<int, 3> restDims = {1, 2, 3};
        shape::TAD xTad(images->getShapeInfo(), restDims.data(), 3);
        xTad.createTadOnlyShapeInfo();
        xTad.createOffsets();

        shape::TAD zTad(output->getShapeInfo(), restDims.data(), 3);
        zTad.createTadOnlyShapeInfo();
        zTad.createOffsets();

        PointersManager manager(context, "helpers::extractPatches");
        auto pxTadShape = (Nd4jLong *) manager.replicatePointer(xTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(xTad.tadOnlyShapeInfo));
        auto pzTadShape = (Nd4jLong *) manager.replicatePointer(zTad.tadOnlyShapeInfo, shape::shapeInfoByteLength(zTad.tadOnlyShapeInfo));
        auto pxTadOffsets = (Nd4jLong *) manager.replicatePointer(xTad.tadOffsets, xTad.numTads * sizeof(Nd4jLong));
        auto pzTadOffsets = (Nd4jLong *) manager.replicatePointer(zTad.tadOffsets, zTad.numTads * sizeof(Nd4jLong));

        int lastDim = images->sizeAt(3);
        int rowDim = images->sizeAt(1);
        int colDim = images->sizeAt(2);

        globalExtractPatches_<T><<<512, 512, 1024, *context->getCudaStream()>>>(images->getSpecialBuffer(), pxTadShape, pxTadOffsets, output->getSpecialBuffer(), pzTadShape, pzTadOffsets, xTad.numTads, sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame, lastDim, rowDim, colDim);

        manager.synchronize();
    }
    BUILD_SINGLE_TEMPLATE(template void _extractPatches, (graph::LaunchContext* context, NDArray* input, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame), LIBND4J_TYPES);



    void extractPatches(graph::LaunchContext* context, NDArray* images, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
        auto xType = images->dataType();

        BUILD_SINGLE_SELECTOR(xType, _extractPatches, (context, images, output, sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame), LIBND4J_TYPES);
    }
}
}
}