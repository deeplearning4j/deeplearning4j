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
#include <ops/declarable/helpers/crop_and_resize.h>
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
void cropAndResizeFunctor_(LaunchContext* context, NDArray * images, NDArray * boxes,
                           NDArray * indices, NDArray * cropSize, int method, double extrapolationVal,
                           NDArray* crops) {
  const int batchSize = images->sizeAt(0);
  const int imageHeight = images->sizeAt(1);
  const int imageWidth = images->sizeAt(2);

  const int numBoxes = crops->sizeAt(0);
  const int cropHeight = crops->sizeAt(1);
  const int cropWidth = crops->sizeAt(2);
  const int depth = crops->sizeAt(3);

  for (auto b = 0; b < numBoxes; ++b) {
    Z y1 = static_cast<Z>(boxes->t<Z>(b, 0));
    Z x1 = static_cast<Z>(boxes->t<Z>(b, 1));
    Z y2 = static_cast<Z>(boxes->t<Z>(b, 2));
    Z x2 = static_cast<Z>(boxes->t<Z>(b, 3));

    int bIn = indices->e<I>(b);
    if (bIn >= batchSize) {
      continue;
    }

    Z heightScale = (cropHeight > 1)
                        ? Z((y2 - y1) * (imageHeight - 1) / (cropHeight - 1))
                        : Z(0);
    Z widthScale = (cropWidth > 1)
                       ? Z((x2 - x1) * (imageWidth - 1) / (cropWidth - 1))
                       : Z(0);

    auto func = PRAGMA_THREADS_FOR {
      for (auto y = start; y < stop; y++) {
        const float inY =
            (cropHeight > 1) ? y1 * (imageHeight - 1) + y * heightScale : 0.5 * (y1 + y2) * (imageHeight - 1);

        if (inY < 0 || inY > imageHeight - 1) {
          for (auto x = 0; x < cropWidth; ++x) {
            for (auto d = 0; d < depth; ++d) {
              crops->p(b, y, x, d, extrapolationVal);
            }
          }
          continue;
        }
        if (method == 0 /* bilinear */) {
          const int topYIndex = sd::math::p_floor(inY);
          const int bottomYIndex = sd::math::p_ceil(inY);
          const float y_lerp = inY - topYIndex;

          for (auto x = 0; x < cropWidth; ++x) {
            const float in_x =
                (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale : 0.5 * (x1 + x2) * (imageWidth - 1);

            if (in_x < 0 || in_x > imageWidth - 1) {
              for (auto d = 0; d < depth; ++d) {
                crops->p(b, y, x, d, extrapolationVal);
              }
              continue;
            }
            int left_x_index = math::p_floor(in_x);
            int right_x_index = math::p_ceil(in_x);
            T x_lerp = static_cast<T>(in_x - left_x_index);

            for (auto d = 0; d < depth; ++d) {
              const T topLeft(images->e<T>(bIn, topYIndex, left_x_index, d));
              const T topRight(images->e<T>(bIn, topYIndex, right_x_index, d));
              const T bottomLeft(images->e<T>(bIn, bottomYIndex, left_x_index, d));
              const T bottomRight(images->e<T>(bIn, bottomYIndex, right_x_index, d));
              const T top = topLeft + (topRight - topLeft) * x_lerp;
              const T bottom = bottomLeft + (bottomRight - bottomLeft) * x_lerp;
              crops->p(b, y, x, d, top + (bottom - top) * y_lerp);
            }
          }
        } else {  // method is "nearest neighbor"
          for (auto x = 0; x < cropWidth; ++x) {
            const float inX =
                (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale : 0.5 * (x1 + x2) * (imageWidth - 1);

            if (inX < 0 || inX > imageWidth - 1) {
              for (auto d = 0; d < depth; ++d) {
                crops->p(b, y, x, d, extrapolationVal);
              }
              continue;
            }
            const int closestXIndex = roundf(inX);
            const int closestYIndex = roundf(inY);
            for (auto d = 0; d < depth; ++d) {
              crops->p(b, y, x, d, images->e<T>(bIn, closestYIndex, closestXIndex, d));
            }
          }
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, cropHeight);
  }
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif