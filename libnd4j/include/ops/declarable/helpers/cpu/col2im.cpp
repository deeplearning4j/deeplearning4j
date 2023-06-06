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
// Created by raver119 on 30.11.17.
//

#include <execution/Threads.h>
#include <ops/declarable/helpers/col2im.h>
#if NOT_EXCLUDED(OP_col2im)

namespace sd {
namespace ops {
namespace helpers {

// [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]
template <typename T>
static void col2im_(sd::LaunchContext& context, const NDArray& input, NDArray& output, const LongType sH, const LongType sW,
                    const LongType pH, const LongType pW, const LongType iH, const LongType iW, const LongType dH, const LongType dW) {
  auto imBuff = output.bufferAsT<T>();
  auto colBuff = input.bufferAsT<T>();
  auto imShapeBuffer = output.shapeInfo();
  auto colShapeBuffer = input.shapeInfo();
  auto colShape = shape::shapeOf(colShapeBuffer);
  auto colStride = shape::stride(colShapeBuffer);
  auto imShape = shape::shapeOf(imShapeBuffer);
  auto imStride = shape::stride(imShapeBuffer);

  const LongType bS = imShape[0];
  const LongType iC = imShape[1];
  const LongType kH = colShape[2];
  const LongType kW = colShape[3];
  const LongType oH = colShape[4];
  const LongType oW = colShape[5];
  const sd::LongType colStride0 = colStride[0];
  const sd::LongType colStride1 = colStride[1];
  const sd::LongType colStride2 = colStride[2];
  const sd::LongType colStride3 = colStride[3];
  const sd::LongType colStride4 = colStride[4];
  const sd::LongType colStride5 = colStride[5];
  const sd::LongType imStride0 = imStride[0];
  const sd::LongType imStride1 = imStride[1];
  const sd::LongType imStride2 = imStride[2];
  const sd::LongType imStride3 = imStride[3];

    auto func = PRAGMA_THREADS_FOR {
      for (auto b = start; b < stop; b++) {
        T* im0 = imBuff + b * imStride0;
        T const* col4 = colBuff + b * colStride0;
        for (sd::LongType colH = 0; colH < oH; ++colH, col4 += colStride4) {
          T const* col5 = col4;
          for (sd::LongType colW = 0; colW < oW; ++colW, col5 += colStride5) {
            T const* col1 = col5;
            T* im1 = im0;
            for (sd::LongType c = 0; c < iC; ++c, col1 += colStride1, im1 += imStride1) {
              sd::LongType imRow = (-pH + colH * sH);
              T const* col2 = col1;
              T* im2 = im1 + imRow * imStride2;
              for (sd::LongType kRow = 0; kRow < kH; ++kRow, col2 += colStride2, imRow += dH, im2 += dH * imStride2) {
                sd::LongType imCol = -pW + colW * sW;
                T const* col3 = col2;
                T* im3 = im2 + imCol * imStride3;
                for (sd::LongType kCol = 0; kCol < kW; ++kCol, col3 += colStride3, imCol += dW, im3 += dW * imStride3) {
                  if (static_cast<LongType>(imRow) < static_cast<LongType>(iH) &&
                      static_cast<LongType>(imRow) >= 0 &&
                      static_cast<LongType>(imCol) < static_cast<LongType>(iW) &&
                      static_cast<LongType>(imCol) >= 0)
                    *im3 += *col3;
                }
              }
            }
          }
        }
      }
    };

    samediff::Threads::parallel_tad(func, 0, bS);

}

void col2im(sd::LaunchContext& context, const NDArray& input, NDArray& output, const LongType sH, const LongType sW, const LongType pH,
            const LongType pW, const LongType iH, const LongType iW, const LongType dH, const LongType dW) {
  BUILD_SINGLE_SELECTOR(input.dataType(), col2im_, (context, input, output, sH, sW, pH, pW, iH, iW, dH, dW),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif