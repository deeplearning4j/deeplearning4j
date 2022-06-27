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
//  @author sgazeos@gmail.com
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/axis.h>
#if NOT_EXCLUDED(OP_extract_patches)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void _extractPatches(NDArray* images, NDArray* output, int sizeRow, int sizeCol, int strideRow, int strideCol,
                            int rateRow, int rateCol, bool theSame) {
  std::vector<int> restDims({1, 2, 3});  // the first and the last dims
  ResultSet listOfMatricies = images->allTensorsAlongDimension(restDims);
  ResultSet listOfOutputs = output->allTensorsAlongDimension(restDims);
  // 3D matricies - 2D matricies of vectors (if last dim is greater than 1)
  // int e = 0;
  const int ksizeRowsEffective = sizeRow + (sizeRow - 1) * (rateRow - 1);
  const int ksizeColsEffective = sizeCol + (sizeCol - 1) * (rateCol - 1);
  const int ksize = ksizeRowsEffective * ksizeColsEffective;
  int batchCount = listOfMatricies.size();  // lengthOf() / ksize;
  sd::LongType lastDim = images->sizeAt(3);
  sd::LongType outLastDim = output->sizeAt(3);
  sd::LongType rowDim = images->sizeAt(1);
  sd::LongType colDim = images->sizeAt(2);
  sd::LongType outRowDim = output->sizeAt(1);
  sd::LongType outColDim = output->sizeAt(2);
  auto rowCast = 1;  //(sizeRow - 1)*rateRow < outRowDim/sizeRow  ?0:1;///(ksize * lastDim > rowDim * ksizeColsEffective
                     //+ lastDim?1:0);
  auto colCast = 1;  // colDim / ksizeColsEffective +2 <= sizeCol?0:1;//(ksize * lastDim > ksizeRowsEffective * colDim +
                     // lastDim?1:0);
  if (sizeRow * rateRow < 3) rowCast = 0;
  if (sizeCol * rateCol < 3) colCast = 0;

  auto func = PRAGMA_THREADS_FOR {
    for (auto batch = 0; batch < stop; batch++) {
      auto patch = listOfMatricies.at(batch);
      auto outMatrix = listOfOutputs.at(batch);

      for (sd::LongType i = 0; i < outRowDim; i++) {
        for (sd::LongType j = 0; j < outColDim; j++) {
          sd::LongType pos = 0;
          // for (sd::LongType k = 0; k < outputLastDim; k++) {
          auto rowStart = i * strideRow - (theSame ? rowCast : 0);
          auto colStart = j * strideCol - (theSame ? colCast : 0);
          auto rowEnd = rowStart + sizeRow * rateRow;
          auto colEnd = colStart + sizeCol * rateCol;
          if (!theSame) {
            rowEnd = math::sd_min(rowStart + sizeRow * rateRow, rowDim);
            colEnd = math::sd_min(colStart + sizeCol * rateCol, colDim);
          }
          // auto pixel = 0LL;
          for (auto row = rowStart; row < rowEnd; row += rateRow)
            for (auto col = colStart; col < colEnd; col += rateCol)
              for (auto pixel = 0; pixel < lastDim; pixel++) {
                bool setUp = (theSame && row >= 0 && col >= 0 && row < rowDim && col < colDim) || (!theSame);
                if (setUp) {
                  outMatrix->r<T>(i, j, pos) = patch->e<T>(row, col, pixel);
                }
                pos++;
              }
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, batchCount);
}

void extractPatches(sd::LaunchContext* context, NDArray* images, NDArray* output, int sizeRow, int sizeCol,
                    int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame) {
  auto xType = images->dataType();

  BUILD_SINGLE_SELECTOR(xType, _extractPatches,
                        (images, output, sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame),
                        SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void _extractPatches,
                      (NDArray * input, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol,
                       int rateRow, int rateCol, bool theSame),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif