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
static void col2im_(sd::LaunchContext& context, const NDArray* input, NDArray* output, const LongType sH, const LongType sW,
                    const LongType pH, const LongType pW, const LongType iH, const LongType iW, const LongType dH, const LongType dW) {
  if(input->rankOf() != 6) {
    THROW_EXCEPTION("ops::helpers::col2im: input array must have rank = 6");
  }

  if(output->rankOf() != 4) {
    THROW_EXCEPTION("ops::helpers::col2im: output array must have rank = 4");
  }

  auto colBuff = input->bufferAsT<T>();
  auto imBuff = output->bufferAsT<T>();
  auto colShapeBuffer = input->shapeInfo();
  auto imShapeBuffer = output->shapeInfo();
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
      LongType im0Offset = b * imStride0;
      LongType col4Offset = b * colStride0;
      for (int colH = 0; colH < oH; ++colH) {
        LongType col5Offset = col4Offset + colH * colStride4;
        for (int colW = 0; colW < oW; ++colW) {
          LongType col1Offset = col5Offset + colW * colStride5;
          LongType im1Offset = im0Offset;
          for (int c = 0; c < iC; ++c) {
            int imRow = (-pH + colH * sH);
            LongType col2Offset = col1Offset + c * colStride1;
            LongType im2Offset = im1Offset + c * imStride1 + imRow * imStride2;
            for (int kRow = 0; kRow < kH; ++kRow) {
              int imCol = -pW + colW * sW;
              LongType col3Offset = col2Offset + kRow * colStride2;
              LongType im3Offset = im2Offset + kRow * dH * imStride2 + imCol * imStride3;
              for (int kCol = 0; kCol < kW; ++kCol) {
                if (static_cast<unsigned>(imRow) < static_cast<unsigned>(iH) &&
                    static_cast<unsigned>(imCol) < static_cast<unsigned>(iW)) {
                  imBuff[im3Offset] += colBuff[col3Offset];
                }
                col3Offset += colStride3;
                imCol += dW;
                im3Offset += dW * imStride3;
              }
              imRow += dH;
            }
          }
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, bS);
}
void col2im(LaunchContext& context, const NDArray* input, NDArray* output, const LongType sH, const LongType sW, const LongType pH,
            const LongType pW, const LongType iH, const LongType iW, const LongType dH, const LongType dW) {
  BUILD_SINGLE_SELECTOR(input->dataType(), col2im_, (context, input, output, sH, sW, pH, pW, iH, iW, dH, dW),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif