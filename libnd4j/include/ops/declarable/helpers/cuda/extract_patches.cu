/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/axis.h>
#include <execution/Threads.h>

#include <array>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extract patches kernel
//      - theSame - SAME or VALID - output format
//      - batchCount - batches - the first dimension of input
//      - sizeRow, sizeCol - rows and cols sizes for batch
//      - rowDim, colDim - rows and cols dimensions for input patches
//      - outRowDim, outColDim - rows and cols dimensions for output patches
//      - strideRow, strideCol - step between input elements with patches
//      - rateRow, rateCol - counts for input patches
//      - rowCast, colCast - shifts for output placement (1 or 0)
//      - lastDim - last dimension of input/output
//      - input - input tensor buffer
//      - patchShape - input patch TAD shape
//      - inputOffsets - input TAD offsets
//      - output - output tensor buffer
//      - outTadShape - output TAD shape
//      - outputOffsets - output TAD offsets
//
template <typename T>
static SD_KERNEL void globalExtractPatchesKernel(bool theSame, int batchCount,
                                                 int sizeRow, int sizeCol, int rowDim,
                                                 int colDim, int outRowDim,
                                                 int outColDim, int strideRow,
                                                 int strideCol,
                                                 int rateRow, int rateCol,
                                                 int rowCast,
                                                 int colCast, int lastDim,
                                                 const T* input,
                                                 const LongType* patchShape,
                                                 const LongType* inputOffsets,
                                                 T* output,
                                                 const LongType* outTadShape,
                                                 const LongType* outputOffsets) {

  for (auto batch = threadIdx.x; batch < batchCount; batch+= gridDim.x) {
    auto patch = input + inputOffsets[batch];
    auto outMatrix = output + outputOffsets[batch];

    for (LongType i = 0; i < outRowDim; i++) {
      for (LongType j = 0; j < outColDim; j++) {
        LongType pos = 0;
        auto rowStart = i * strideRow - (theSame ? rowCast : 0);
        auto colStart = j * strideCol - (theSame ? colCast : 0);
        auto rowEnd = rowStart + sizeRow * rateRow;
        auto colEnd = colStart + sizeCol * rateCol;
        if (!theSame) {
          rowEnd = math::sd_min<T>(rowStart + sizeRow * rateRow, rowDim);
          colEnd = math::sd_min<T>(colStart + sizeCol * rateCol, colDim);
        }
        for (auto row = rowStart; row < rowEnd; row += rateRow)
          for (auto col = colStart; col < colEnd; col += rateCol)
            for (auto pixel = 0; pixel < lastDim; pixel++) {
              LongType zPos[] = {i, j, pos};
              LongType xPos[] = {row, col, pixel};
              bool setUp = (theSame && row >= 0 && col >= 0 && row < rowDim && col < colDim) || (!theSame);

              if (setUp) {  // VALID or SAME cases
                outMatrix[shape::getOffset(outTadShape, zPos)] = patch[shape::getOffset(patchShape, xPos)];
              }
              pos++;
            }
      }
    }
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void _extractPatches(LaunchContext* context, NDArray* images, NDArray* output, int sizeRow, int sizeCol,
                            int strideRow, int strideCol, int rateRow, int rateCol, bool theSame) {
  std::vector<LongType> restDims({1, 2, 3});  // the first and the last dims
  ResultSet listOfMatricies = images->allTensorsAlongDimension(restDims);
  ResultSet listOfOutputs = output->allTensorsAlongDimension(restDims);
  // 3D matrices - 2D matrices of vectors (if last dim is greater than 1)
  // int e = 0;

  int batchCount = listOfMatricies.size();
  LongType lastDim = images->sizeAt(3);
  LongType rowDim = images->sizeAt(1);
  LongType colDim = images->sizeAt(2);
  LongType outRowDim = output->sizeAt(1);
  LongType outColDim = output->sizeAt(2);
  auto rowCast = 1;
  auto colCast = 1;
  if (sizeRow * rateRow < 3) rowCast = 0;
  if (sizeCol * rateCol < 3) colCast = 0;

  auto func = PRAGMA_THREADS_FOR {
    for (auto batch = 0; batch < stop; batch++) {
      auto patch = listOfMatricies.at(batch);
      auto inPatch = patch->rankOf() > 3 && patch->sizeAt(0) == 1 ? new NDArray(patch->reshape('c',{patch->sizeAt(1),patch->sizeAt(2),patch->sizeAt(3)})) : patch;
      auto outMatrix = listOfOutputs.at(batch);
      auto outReshape = outMatrix->rankOf() > 3 && outMatrix->sizeAt(0) == 1 ? new NDArray(outMatrix->reshape('c',{outMatrix->sizeAt(1),outMatrix->sizeAt(2),outMatrix->sizeAt(3)})) : outMatrix;
      for (LongType i = 0; i < outRowDim; i++) {
        for (LongType j = 0; j < outColDim; j++) {
          LongType pos = 0;
          auto rowStart = i * strideRow - (theSame ? rowCast : 0);
          auto colStart = j * strideCol - (theSame ? colCast : 0);
          auto rowEnd = rowStart + sizeRow * rateRow;
          auto colEnd = colStart + sizeCol * rateCol;
          if (!theSame) {
            rowEnd = math::sd_min(rowStart + sizeRow * rateRow, rowDim);
            colEnd = math::sd_min(colStart + sizeCol * rateCol, colDim);
          }
          for (auto row = rowStart; row < rowEnd; row += rateRow)
            for (auto col = colStart; col < colEnd; col += rateCol)
              for (auto pixel = 0; pixel < lastDim; pixel++) {
                bool setUp = (theSame && row >= 0
                              && col >= 0 && row < rowDim
                              && col < colDim)
                             || (!theSame);
                if (setUp) {
                  outReshape->p<T>(i,j,pos,inPatch->e<T>(row, col, pixel));
                }
                pos++;
              }
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, batchCount);
}
BUILD_SINGLE_TEMPLATE(template void _extractPatches,
                      (sd::LaunchContext * context, NDArray* input, NDArray* output, int sizeRow, int sizeCol,
                          int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame),
                      SD_COMMON_TYPES);

void extractPatches(LaunchContext* context, NDArray* images, NDArray* output, int sizeRow, int sizeCol,
                    int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame) {
  auto xType = images->dataType();

  BUILD_SINGLE_SELECTOR(xType, _extractPatches,
                        (context, images, output, sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame),
                        SD_COMMON_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
