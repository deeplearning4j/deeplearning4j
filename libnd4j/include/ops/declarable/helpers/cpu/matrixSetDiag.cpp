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

  // Cache all shape information upfront
  const sd::LongType* xShapeInfo = input.shapeInfo();
  const sd::LongType* yShapeInfo = diagonal.shapeInfo();
  const sd::LongType* zShapeInfo = output.shapeInfo();

  // Cache shape-related values
  const int xRank = input.rankOf();
  const auto xLen = input.lengthOf();

  // Cache shape and stride pointers
  const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
  const sd::LongType* xStride = shape::stride(xShapeInfo);
  const sd::LongType* yStride = shape::stride(yShapeInfo);
  const sd::LongType* zStride = shape::stride(zShapeInfo);

  // Check if input and output have same offsets
  const bool areSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    // Pre-allocate coords array outside the loop
    sd::LongType coords[SD_MAX_RANK];

    for (sd::LongType i = 0; i < xLen; ++i) {
      // Use cached shape data for coordinate transforms
      INDEX2COORDS(i, xRank, xShape, coords);

      sd::LongType xOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);

      sd::LongType zOffset;
      if (areSameOffsets) {
        zOffset = xOffset;
      } else {
        COORDS2INDEX(xRank, zStride, coords, zOffset);
      }

      // Check diagonal condition using cached rank
      if (coords[xRank - 2] == coords[xRank - 1]) {
        sd::LongType yOffset;
        COORDS2INDEX(xRank - 1, yStride, coords, yOffset);
        z[zOffset] = y[yOffset];
      } else {
        z[zOffset] = zeroPad ? static_cast<T>(0) : x[xOffset];
      }
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