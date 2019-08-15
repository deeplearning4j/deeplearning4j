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

//    template <typename T>
//    static __global__ void globalExtractPatchesKernel(bool theSame, int batchCount, int sizeRow, int sizeCol, int rowDim, int colDim, int outRowDim, int outColDim, int strideRow, int strideCol, int rateRow, int rateCol, int rowCast, int colCast, int lastDim, T* input, Nd4jLong* patchShape, Nd4jLong* inputOffsets, T* output, Nd4jLong* outTadShape, Nd4jLong* outputOffsets) {
//    //globalExtractPatches_(void *vinput, Nd4jLong *xTadShape, Nd4jLong *xTadOffsets, void *voutput, Nd4jLong *zTadShape, Nd4jLong *zTadOffsets, const int numTads, const int sizeRow, const int sizeCol, const int stradeRow, const int stradeCol, const int rateRow, const int rateCol, const bool theSame, const int lastDim, const int rowDim, const int colDim) {
//        const int warpSize = lastDim;
//        const int tid = blockIdx.x * gridDim.x + threadIdx.x;
//        const int warpIdx = tid / warpSize;
//        const int warpPos = tid % warpSize;
//        const int numWarps = 1; //(gridDim.x * blockDim.x) / warpSize;
//        const int patchLength = shape::length(outTadShape);
//
//        auto xShape = shape::shapeOf(patchShape);
//        auto xStride = shape::stride(patchShape);
//        auto xRank = shape::rank(patchShape);
//
//        for (int e = 0; e < batchCount; e += numWarps) {
//            auto patch = input + inputOffsets[e];
//            auto matrix = output + outputOffsets[e];
//            int iter = 0;
//
//            for (Nd4jLong i = 0; i < outRowDim; i++) {
//                for (Nd4jLong j = 0; j < outColDim; j++) {
//                    Nd4jLong pos = 0;
//                    //for (Nd4jLong k = 0; k < outputLastDim; k++) {
//                    auto rowStart = i * strideRow - (theSame?rowCast:0);
//                    auto colStart = j * strideCol - (theSame?colCast:0);
//                    auto rowEnd = rowStart + sizeRow * rateRow;
//                    auto colEnd = colStart + sizeCol * rateCol;
//                    if (!theSame) {
//                        rowEnd = math::nd4j_min(int(rowStart + sizeRow * rateRow), rowDim);
//                        colEnd = math::nd4j_min(int(colStart + sizeCol * rateCol), colDim);
//                    }
//                    //auto pixel = 0LL;
//                    for (auto row = rowStart; row < rowEnd; row += rateRow)
//                        for (auto col = colStart; col < colEnd; col += rateCol)
//                            for (auto pixel = 0; pixel < lastDim; pixel++) {
//                                Nd4jLong zPos[] = {i, j, pos};
//                                Nd4jLong xPos[] = {row, col, pixel};
//                                auto zIndex = shape::getOffset(0, shape::shapeOf(outTadShape), shape::stride(outTadShape), zPos, 3);
//                                auto xIndex = shape::getOffset(0, shape::shapeOf(patchShape), shape::stride(patchShape), xPos, 3);
//                                if (theSame) { // SAME case
//                                    if (row >= 0 && col >= 0 && row < rowDim && col < colDim)
//                                        matrix[zIndex] = patch[xIndex]; //outMatrix->p<T>(i, j, pos, patch->e<T>(row, col, pixel));
//                                    //pos++;
//                                }
//                                else { // VALID case
//                                    matrix[zIndex] = patch[xIndex]; //outMatrix->p<T>(i, j, pos++, patch->e<T>(row, col, pixel));
//                                }
//                                pos++;
//                            }
//                }
//            }
//            __syncthreads();
//        }
//    }

    template <typename T>
    static __global__ void globalExtractPatchesKernel(bool theSame, int batchCount, int sizeRow, int sizeCol, int rowDim, int colDim, int outRowDim, int outColDim, int strideRow, int strideCol, int rateRow, int rateCol, int rowCast, int colCast, int lastDim, T* input, Nd4jLong* patchShape, Nd4jLong* inputOffsets, T* output, Nd4jLong* outTadShape, Nd4jLong* outputOffsets) {
        __shared__ Nd4jLong* xShapeOf;
        __shared__ Nd4jLong* xStrideOf;
        __shared__ Nd4jLong* zShapeOf;
        __shared__ Nd4jLong* zStrideOf;

        if (0 == threadIdx.x) {
            xShapeOf = shape::shapeOf(patchShape);
            xStrideOf = shape::stride(patchShape);
            zShapeOf = shape::shapeOf(outTadShape);
            zStrideOf = shape::stride(outTadShape);
        }
        __syncthreads();

        auto start = threadIdx.x + blockIdx.x * blockDim.x;

        auto step = blockDim.x * gridDim.x;

        for (Nd4jLong batch = start; batch < batchCount; batch += step) {
            auto patch = input + inputOffsets[batch];// listOfMatricies->at(batch);
            auto outMatrix = output + outputOffsets[batch]; //listOfOutputs->at(batch);

            for (Nd4jLong i = 0; i < outRowDim; i++) {
                for (Nd4jLong j = 0; j < outColDim; j++) {
                    Nd4jLong pos = 0;
                    //for (Nd4jLong k = 0; k < outputLastDim; k++) {
                    auto rowStart = i * strideRow - (theSame?rowCast:0);
                    auto colStart = j * strideCol - (theSame?colCast:0);
                    auto rowEnd = rowStart + sizeRow * rateRow;
                    auto colEnd = colStart + sizeCol * rateCol;
                    if (!theSame) {
                        rowEnd = math::nd4j_min(rowStart + sizeRow * rateRow, Nd4jLong (rowDim));
                        colEnd = math::nd4j_min(colStart + sizeCol * rateCol, Nd4jLong (colDim));
                    }
                    //auto pixel = 0LL;
                    for (auto row = rowStart; row < rowEnd; row += rateRow)
                        for (auto col = colStart; col < colEnd; col += rateCol)
                            for (auto pixel = 0; pixel < lastDim; pixel++) {
                                Nd4jLong zPos[] = {i, j, pos};
                                Nd4jLong xPos[] = {row, col, pixel};
                                bool setUp = (theSame && row >= 0 && col >= 0 && row < rowDim && col < colDim) || (!theSame);

                                if (setUp) { // VALID or SAME cases
                                    outMatrix[shape::getOffset(0, zShapeOf, zStrideOf, zPos, 3)] = patch[shape::getOffset(0, xShapeOf, xStrideOf, xPos, 3)];
                                }
                                pos++;
                            }
                }
            }
        }

    }

    template <typename T>
    static void _extractPatches(nd4j::LaunchContext * context, NDArray* images, NDArray* output, int sizeRow, int sizeCol, int strideRow, int strideCol, int rateRow, int rateCol, bool theSame){
        NDArray::prepareSpecialUse({output}, {images});
        std::vector<int> restDims({1, 2, 3}); // the first and the last dims
        // 3D matricies - 2D matricies of vectors (if last dim is greater than 1)
        //int e = 0;
        const int ksizeRowsEffective = sizeRow + (sizeRow - 1) * (rateRow - 1);
        const int ksizeColsEffective = sizeCol + (sizeCol - 1) * (rateCol - 1);
        const int ksize = ksizeRowsEffective * ksizeColsEffective;
        Nd4jLong lastDim = images->sizeAt(3);
        Nd4jLong outLastDim = output->sizeAt(3);
        Nd4jLong rowDim = images->sizeAt(1);
        Nd4jLong colDim = images->sizeAt(2);
        Nd4jLong outRowDim = output->sizeAt(1);
        Nd4jLong outColDim = output->sizeAt(2);
        auto rowCast = 1; //(sizeRow - 1)*rateRow < outRowDim/sizeRow  ?0:1;///(ksize * lastDim > rowDim * ksizeColsEffective + lastDim?1:0);
        auto colCast = 1; //colDim / ksizeColsEffective +2 <= sizeCol?0:1;//(ksize * lastDim > ksizeRowsEffective * colDim + lastDim?1:0);
        if (sizeRow * rateRow < 3)
            rowCast = 0;
        if (sizeCol * rateCol < 3)
            colCast = 0;
        //images->tickReadDevice();
        //if (images->isActualOnDeviceSide())
        //images->syncToDevice();

        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(images->getShapeInfo(), restDims.data(), restDims.size());
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), restDims.data(), restDims.size());
        int batchCount = packX.numberOfTads(); //'listOfMatricies->size(); //lengthOf() / ksize;
        //printf("Batch Count is %d\n", batchCount);
        //shape::printShapeInfo(packX.primaryShapeInfo());
        //NDArray::prepareSpecialUse({output}, {images});
        PointersManager manager(context, "helpers::extractPatches");

        auto stream = context->getCudaStream();
        auto imagesBuffer = reinterpret_cast<T*>(images->specialBuffer());
        auto outputBuffer = reinterpret_cast<T*>(output->specialBuffer());
        //images->printIndexedBuffer("INPUT");
//        globalExtractPatchesKernel<T><<<512, 512, 1024, *context->getCudaStream()>>>(theSame, batchCount, sizeRow, sizeCol,
        globalExtractPatchesKernel<T><<<128, 128, 1024, *stream>>>(theSame, batchCount, sizeRow, sizeCol,
                rowDim, colDim, outRowDim, outColDim, strideRow, strideCol, rateRow, rateCol, rowCast, colCast, lastDim,
                imagesBuffer, packX.specialShapeInfo(), packX.specialOffsets(), outputBuffer, packZ.specialShapeInfo(),
                packZ.specialOffsets());
        //extractPatchesKernel<T><<<batchCount, 512, 1024, *stream>>>(theSame, batchCount, sizeRow, sizeCol, rowDim, colDim, outRowDim, outColDim, stradeRow, stradeCol, rateRow, rateCol, rowCast, colCast, lastDim, imagesBuffer, packX.specialShapeInfo(), packX.platformOffsets(), outputBuffer, packZ.specialShapeInfo(), packZ.platformOffsets());
        //output->tickWriteDevice();
        //output->printIndexedBuffer("OUTPUT");
        manager.synchronize();
        NDArray::registerSpecialUse({output}, {images});
    }
    BUILD_SINGLE_TEMPLATE(template void _extractPatches, (nd4j::LaunchContext * context, NDArray* input, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame), LIBND4J_TYPES);



    void extractPatches(nd4j::LaunchContext * context, NDArray* images, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
        auto xType = images->dataType();

        BUILD_SINGLE_SELECTOR(xType, _extractPatches, (context, images, output, sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame), LIBND4J_TYPES);
    }
//        std::vector<int> restDims({1, 2, 3}); // the first and the last dims
//        std::unique_ptr<ResultSet> listOfMatricies(images->allTensorsAlongDimension(restDims));
//        std::unique_ptr<ResultSet> listOfOutputs(output->allTensorsAlongDimension(restDims));
//        // 3D matricies - 2D matricies of vectors (if last dim is greater than 1)
//        //int e = 0;
//        const int ksizeRowsEffective = sizeRow + (sizeRow - 1) * (rateRow - 1);
//        const int ksizeColsEffective = sizeCol + (sizeCol - 1) * (rateCol - 1);
//        const int ksize = ksizeRowsEffective * ksizeColsEffective;
//        int batchCount = listOfMatricies->size(); //lengthOf() / ksize;
//        Nd4jLong lastDim = images->sizeAt(3);
//        Nd4jLong outLastDim = output->sizeAt(3);
//        Nd4jLong rowDim = images->sizeAt(1);
//        Nd4jLong colDim = images->sizeAt(2);
//        Nd4jLong outRowDim = output->sizeAt(1);
//        Nd4jLong outColDim = output->sizeAt(2);
//        auto rowCast = 1; //(sizeRow - 1)*rateRow < outRowDim/sizeRow  ?0:1;///(ksize * lastDim > rowDim * ksizeColsEffective + lastDim?1:0);
//        auto colCast = 1; //colDim / ksizeColsEffective +2 <= sizeCol?0:1;//(ksize * lastDim > ksizeRowsEffective * colDim + lastDim?1:0);
//        if (sizeRow * rateRow < 3)
//            rowCast = 0;
//        if (sizeCol * rateCol < 3)
//            colCast = 0;
//        //Nd4jLong outputLastDim = output->sizeAt(3);
//       PRAGMA_OMP_PARALLEL_FOR
//        for (Nd4jLong batch = 0; batch < batchCount; batch++) {
//            auto patch = listOfMatricies->at(batch);
//            auto outMatrix = listOfOutputs->at(batch);
//            //auto patchBorder = patch->sizeAt(0);
//            if (theSame) { // SAME case
//                for (Nd4jLong i = 0; i < outRowDim; i++) {
//                    for (Nd4jLong j = 0; j < outColDim; j++) {
//                        Nd4jLong pos = 0;
//                        //for (Nd4jLong k = 0; k < outputLastDim; k++) {
//                        auto rowStart = i * strideRow - rowCast;
//                        auto colStart = j * strideCol - colCast;
//                        auto rowEnd = rowStart + sizeRow * rateRow;
//                        auto colEnd = colStart + sizeCol * rateCol;
//                        auto pixel = 0LL;
//                        for (auto row = rowStart; row < rowEnd; row += rateRow)
//                            for (auto col = colStart; col < colEnd; col += rateCol)
//                                for (auto pixel = 0; pixel < lastDim; pixel++) {
//                                    if (row >=0 && col >= 0 && row < rowDim && col < colDim)
//                                    outMatrix->p<T>(i, j, pos, patch->e<T>(row, col, pixel));
//                                    pos++;
//                                }
//                        //}
//                    }
//                }
//
//            } else { // VALID case
//                for (Nd4jLong i = 0; i < outRowDim; i++) {
//                    for (Nd4jLong j = 0; j < outColDim; j++) {
//                        Nd4jLong pos = 0;
//                        //for (Nd4jLong k = 0; k < outputLastDim; k++) {
//                            auto rowStart = i * strideRow;
//                            auto colStart = j * strideCol;
//                            auto rowEnd = math::nd4j_min(rowStart + sizeRow * rateRow, rowDim);
//                            auto colEnd = math::nd4j_min(colStart + sizeCol * rateCol, colDim);
//                            auto pixel = 0LL;
//                            for (auto row = rowStart; row < rowEnd; row += rateRow)
//                                for (auto col = colStart; col < colEnd; col += rateCol)
//                                    for (auto pixel = 0; pixel < lastDim; pixel++)
//                                        outMatrix->p<T>(i,j,pos++, patch->e<T>(row, col, pixel));
//                        //}
//                    }
//                }
//            }
//        }
//
//
//
}
}
}