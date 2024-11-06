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
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <array/ResultSet.h>
#include <execution/Threads.h>
#include <ops/declarable/helpers/matrixSetDiag.h>
#if NOT_EXCLUDED(OP_matrix_set_diag)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
void matrixSetDiag_(NDArray& input, NDArray& diagonal, NDArray& output, const bool zeroPad) {
  // input and output are the same array (x == z) when zeroPad = true
  // xRank = zRank, xRank = yRank + 1
  // xLen = zLen

  const T* x = input.bufferAsT<T>();
  const T* y = diagonal.bufferAsT<T>();
  T* z = output.bufferAsT<T>();

  const sd::LongType* xShapeInfo = input.shapeInfo();
  const sd::LongType* yShapeInfo = diagonal.shapeInfo();
  const sd::LongType* zShapeInfo = output.shapeInfo();

  const bool areSameOffsets =
      shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);  // shapes are definitely the same, but strides might not

  const int xRank = input.rankOf();
  const auto xLen = input.lengthOf();

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];

    for (sd::LongType i = 0; i < xLen; ++i) {
      shape::index2coordsCPU(start, i, xShapeInfo, coords);

      const auto xOffset = shape::getOffset(xShapeInfo, coords);
      const auto zOffset = areSameOffsets ? xOffset : shape::getOffset(zShapeInfo, coords);

      // condition to be on diagonal of innermost matrix
      if (coords[xRank - 2] == coords[xRank - 1])
        z[zOffset] = y[shape::getOffset(yShapeInfo, coords)];
      else
        z[zOffset] = zeroPad ? static_cast<T>(0) : x[xOffset];
    }
  };
  samediff::Threads::parallel_for(func, 0, xLen);
}

//////////////////////////////////////////////////////////////////////////
void matrixSetDiag(sd::LaunchContext* context, NDArray& input, NDArray& diagonal, NDArray& output,
                   const bool zeroPad) {
  BUILD_SINGLE_SELECTOR(input.dataType(), matrixSetDiag_, (input, diagonal, output, zeroPad), SD_COMMON_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif