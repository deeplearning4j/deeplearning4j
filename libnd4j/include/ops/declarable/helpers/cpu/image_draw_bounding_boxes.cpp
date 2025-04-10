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
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//
//  @author sgazeos@gmail.com
//
#include <array/NDArray.h>
#include <execution/Threads.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {
typedef std::vector<std::vector<float>> ColorTable_t;
static ColorTable_t DefaultColorTable(int depth) {
  std::vector<std::vector<float>> colorTable;
  colorTable.emplace_back(std::vector<float>({1, 1, 0, 1}));      // 0: yellow
  colorTable.emplace_back(std::vector<float>({0, 0, 1, 1}));      // 1: blue
  colorTable.emplace_back(std::vector<float>({1, 0, 0, 1}));      // 2: red
  colorTable.emplace_back(std::vector<float>({0, 1, 0, 1}));      // 3: lime
  colorTable.emplace_back(std::vector<float>({0.5, 0, 0.5, 1}));  // 4: purple
  colorTable.emplace_back(std::vector<float>({0.5, 0.5, 0, 1}));  // 5: olive
  colorTable.emplace_back(std::vector<float>({0.5, 0, 0, 1}));    // 6: maroon
  colorTable.emplace_back(std::vector<float>({0, 0, 0.5, 1}));    // 7: navy blue
  colorTable.emplace_back(std::vector<float>({0, 1, 1, 1}));      // 8: aqua
  colorTable.emplace_back(std::vector<float>({1, 0, 1, 1}));      // 9: fuchsia

  if (depth == 1) {
    for (size_t i = 0; i < colorTable.size(); i++) {
      colorTable[i][0] = 1;
    }
  }
  return colorTable;
}

void drawBoundingBoxesFunctor(sd::LaunchContext* context, NDArray* images, NDArray* boxes, NDArray* colors,
                              NDArray* output) {
  // images - batch of 3D images with BW (last dim = 1), RGB (last dim = 3) or RGBA (last dim = 4) channel set
  // boxes - batch of 2D bounds with last dim (y_start, x_start, y_end, x_end) to compute i and j as
  // floor((height - 1 ) * y_start) => rowStart, floor((height - 1) * y_end) => rowEnd
  // floor((width - 1 ) * x_start) => colStart, floor((width - 1) * x_end) => colEnd
  // height = images->sizeAt(1), width = images->sizeAt(2)
  // colors - colors for each box given
  // set up color for each box as frame
  auto batchSize = images->sizeAt(0);
  auto boxSize = boxes->sizeAt(0);
  auto height = images->sizeAt(1);
  auto width = images->sizeAt(2);
  auto channels = images->sizeAt(3);

  output->assign(images);  // fill up all output with input images, then fill up boxes
  ColorTable_t colorTable;
  if (colors) {
    for (auto i = 0; i < colors->sizeAt(0); i++) {
      std::vector<float> colorValue(4);
      for (auto j = 0; j < 4; j++) {
        colorValue[j] = j < colors->sizeAt(1) ? colors->e<float>(i, j) : 1.f;
      }
      colorTable.emplace_back(colorValue);
    }
  }
  if (colorTable.empty()) colorTable = DefaultColorTable(channels);
  auto func = PRAGMA_THREADS_FOR {
    for (auto batch = start; batch < stop; ++batch) {  // loop by batch
      const sd::LongType numBoxes = boxes->sizeAt(1);
      for (auto boxIndex = 0; boxIndex < numBoxes; ++boxIndex) {
        auto colorIndex = boxIndex % colorTable.size();
        auto rowStart = sd::LongType((height - 1) * boxes->t<float>(batch, boxIndex, 0));
        auto rowStartBound = sd::math::sd_max(sd::LongType(0), rowStart);
        auto rowEnd = sd::LongType((height - 1) * boxes->t<float>(batch, boxIndex, 2));
        auto rowEndBound = sd::math::sd_min(sd::LongType(height - 1), rowEnd);
        auto colStart = sd::LongType((width - 1) * boxes->t<float>(batch, boxIndex, 1));
        auto colStartBound = sd::math::sd_max(sd::LongType(0), colStart);
        auto colEnd = sd::LongType((width - 1) * boxes->t<float>(batch, boxIndex, 3));
        auto colEndBound = sd::math::sd_min(sd::LongType(width - 1), colEnd);

        if (rowStart > rowEnd || colStart > colEnd) {
          sd_debug(
              "helpers::drawBoundingBoxesFunctor: Bounding box (%lld, %lld, %lld, %lld) is inverted "
              "and will not be drawn\n",
              rowStart, colStart, rowEnd, colEnd);
          continue;
        }
        if (rowStart >= height || rowEnd < 0 || colStart >= width || colEnd < 0) {
          sd_debug(
              "helpers::drawBoundingBoxesFunctor: Bounding box (%lld, %lld, %lld, %lld) is completely "
              "outside the image and not be drawn\n ",
              rowStart, colStart, rowEnd, colEnd);
          continue;
        }

        // Draw upper line
        if (rowStart >= 0) {
          for (auto j = colStartBound; j <= colEndBound; ++j)
            for (auto c = 0; c < channels; c++) {
              output->p(batch, rowStart, j, c, colorTable[colorIndex][c]);
            }
        }
        // Draw bottom line.
        if (rowEnd < height) {
          for (auto j = colStartBound; j <= colEndBound; ++j)
            for (auto c = 0; c < channels; c++) {
              output->p(batch, rowEnd, j, c, colorTable[colorIndex][c]);
            }
        }

        // Draw left line.
        if (colStart >= 0) {
          for (auto i = rowStartBound; i <= rowEndBound; ++i)
            for (auto c = 0; c < channels; c++) {
              output->p(batch, i, colStart, c, colorTable[colorIndex][c]);
            }
        }
        // Draw right line.
        if (colEnd < width) {
          for (auto i = rowStartBound; i <= rowEndBound; ++i)
            for (auto c = 0; c < channels; c++) {
              output->p(batch, i, colEnd, c, colorTable[colorIndex][c]);
            }
        }
      }
    }
  };
  samediff::Threads::parallel_tad(func, 0, batchSize);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
