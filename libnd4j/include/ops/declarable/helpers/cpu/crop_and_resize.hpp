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
#pragma once

//
//  @author sgazeos@gmail.com
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/crop_and_resize.h>
#include <ops/declarable/helpers/image_resize.h>

#include <type_traits>
#if NOT_EXCLUDED(OP_crop_and_resize)
namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cropAndResizeFunctor main algorithm
//      context - launch context
//      images - batch of images (4D tensor - [batch, width, height, pixels])
//      boxes - 2D tensor with boxes for crop
//      indices - 2D int tensor with indices of boxes to crop
//      cropSize - 2D int tensor with crop box sizes
//      method - (one of 0 - bilinear, 1 - nearest)
//      extrapolationVal - double value of extrapolation
//      crops - output (4D tensor - [batch, outWidth, outHeight, pixels])
//
template <typename T, typename Z, typename I>
SD_LIB_EXPORT void cropAndResizeFunctor_(LaunchContext* context, NDArray * images, NDArray * boxes,
                           NDArray * indices, NDArray * cropSize, int method, double extrapolationVal,
                           NDArray* crops) {
  // the sample positions and the interpolation are computed in double when the images or the boxes are DOUBLE, and in
  // float otherwise (a float position limited a DOUBLE crop to float precision, and an integer image's weights to 0)
  using PosT = typename std::conditional<std::is_same<T, double>::value || std::is_same<Z, double>::value, double,
                                         float>::type;

  const LongType batchSize = images->sizeAt(0);
  const LongType imageHeight = images->sizeAt(1);
  const LongType imageWidth = images->sizeAt(2);

  const LongType numBoxes = crops->sizeAt(0);
  const LongType cropHeight = crops->sizeAt(1);
  const LongType cropWidth = crops->sizeAt(2);
  const LongType depth = crops->sizeAt(3);

  for (LongType b = 0; b < numBoxes; ++b) {
    const PosT y1 = static_cast<PosT>(boxes->t<Z>(b, 0));
    const PosT x1 = static_cast<PosT>(boxes->t<Z>(b, 1));
    const PosT y2 = static_cast<PosT>(boxes->t<Z>(b, 2));
    const PosT x2 = static_cast<PosT>(boxes->t<Z>(b, 3));

    // a box that names no image of the batch is outside every image: its crop is the extrapolation value (it was left
    // unwritten, and an index below zero read before the images)
    const LongType bIn = indices->e<I>(b);
    const bool inBatch = bIn >= 0 && bIn < batchSize;

    const PosT heightScale = cropResizeScale<PosT>(y1, y2, imageHeight, cropHeight);
    const PosT widthScale = cropResizeScale<PosT>(x1, x2, imageWidth, cropWidth);

    auto func = PRAGMA_THREADS_FOR {
      for (auto y = start; y < stop; y++) {
        const PosT inY = cropResizeCoordinate<PosT>(y1, y2, imageHeight, cropHeight, y, heightScale);

        if (!inBatch || !(inY >= 0 && inY <= imageHeight - 1)) {
          for (LongType x = 0; x < cropWidth; ++x) {
            for (LongType d = 0; d < depth; ++d) {
              crops->p(b, y, x, d, extrapolationVal);
            }
          }
          continue;
        }
        if (method == 0 /* bilinear */) {
          const LongType topYIndex = static_cast<LongType>(sd::math::p_floor<PosT>(inY));
          const LongType bottomYIndex = static_cast<LongType>(sd::math::p_ceil<PosT>(inY));
          const PosT yLerp = inY - topYIndex;

          for (LongType x = 0; x < cropWidth; ++x) {
            const PosT inX = cropResizeCoordinate<PosT>(x1, x2, imageWidth, cropWidth, x, widthScale);

            if (!(inX >= 0 && inX <= imageWidth - 1)) {
              for (LongType d = 0; d < depth; ++d) {
                crops->p(b, y, x, d, extrapolationVal);
              }
              continue;
            }
            const LongType leftXIndex = static_cast<LongType>(sd::math::p_floor<PosT>(inX));
            const LongType rightXIndex = static_cast<LongType>(sd::math::p_ceil<PosT>(inX));
            const PosT xLerp = inX - leftXIndex;

            for (LongType d = 0; d < depth; ++d) {
              const PosT topLeft = static_cast<PosT>(images->e<T>(bIn, topYIndex, leftXIndex, d));
              const PosT topRight = static_cast<PosT>(images->e<T>(bIn, topYIndex, rightXIndex, d));
              const PosT bottomLeft = static_cast<PosT>(images->e<T>(bIn, bottomYIndex, leftXIndex, d));
              const PosT bottomRight = static_cast<PosT>(images->e<T>(bIn, bottomYIndex, rightXIndex, d));
              const PosT top = imageResizeLerp<PosT>(topLeft, topRight, xLerp);
              const PosT bottom = imageResizeLerp<PosT>(bottomLeft, bottomRight, xLerp);
              crops->p(b, y, x, d, imageResizeLerp<PosT>(top, bottom, yLerp));
            }
          }
        } else {  // method is "nearest neighbor"
          for (LongType x = 0; x < cropWidth; ++x) {
            const PosT inX = cropResizeCoordinate<PosT>(x1, x2, imageWidth, cropWidth, x, widthScale);

            if (!(inX >= 0 && inX <= imageWidth - 1)) {
              for (LongType d = 0; d < depth; ++d) {
                crops->p(b, y, x, d, extrapolationVal);
              }
              continue;
            }
            const LongType closestXIndex = static_cast<LongType>(sd::math::p_round<PosT>(inX));
            const LongType closestYIndex = static_cast<LongType>(sd::math::p_round<PosT>(inY));
            for (LongType d = 0; d < depth; ++d) {
              crops->p(b, y, x, d, images->e<T>(bIn, closestYIndex, closestXIndex, d));
            }
          }
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, cropHeight);
  }
}
}
}  // namespace ops
}  // namespace sd


BUILD_TRIPLE_TEMPLATE(void sd::ops::helpers::cropAndResizeFunctor_,
                      (sd::LaunchContext * context, NDArray * images, NDArray * boxes, NDArray * indices,
                       NDArray * cropSize, int method, double extrapolationVal, NDArray* crops),
                      SD_NUMERIC_TYPES, SD_FLOAT_TYPES, SD_INTEGER_TYPES);
                      
#endif