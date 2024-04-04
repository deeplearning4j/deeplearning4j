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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.09.2018
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/im2col.h>
#if NOT_EXCLUDED(OP_im2col)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void im2col_(sd::LaunchContext& context, const NDArray& input, NDArray& output, const LongType kH, const LongType kW,
                    const LongType sH, const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW,
                    const NDArray& arrZeroPadVal) {
  // input [bS, iC, iH, iW] is convoluted to output [bS, iC, kH, kW, oH, oW]

  auto imBuff = static_cast<T const*>(input.buffer());
  auto colBuff = static_cast<T*>(output.buffer());
  auto imShapeBuffer = input.shapeInfo();
  auto colShapeBuffer = output.shapeInfo();
  auto colShape = shape::shapeOf(colShapeBuffer);
  auto colStride = shape::stride(colShapeBuffer);
  auto imShape = shape::shapeOf(imShapeBuffer);
  auto imStride = shape::stride(imShapeBuffer);

  const T zeroPadVal = arrZeroPadVal.e<T>(0);

  const LongType bS = imShape[0];
  const LongType iC = imShape[1];
  const LongType iH = imShape[2];
  const LongType iW = imShape[3];
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

  auto func = PRAGMA_THREADS_FOR_2D {
    T* col;
    T const* im;
    sd::LongType imRow, imCol;

    for (auto b = start_x; b < stop_x; b += inc_x) {
      for (auto colH = start_y; colH < stop_y; colH += inc_y) {
        for (sd::LongType colW = 0; colW < oW; ++colW) {
          for (sd::LongType c = 0; c < iC; ++c) {
            for (sd::LongType kRow = 0; kRow < kH; ++kRow) {
              for (sd::LongType kCol = 0; kCol < kW; ++kCol) {
                imRow = (-pH + kRow * dH) + colH * sH;
                imCol = (-pW + kCol * dW) + colW * sW;

                col = colBuff + b * colStride0 + c * colStride1 + kRow * colStride2 + kCol * colStride3 +
                      colH * colStride4 + colW * colStride5;
                if (static_cast<LongType>(imRow) >= static_cast<LongType>(iH) ||
                    static_cast<LongType>(imRow) < 0 ||
                    static_cast<LongType>(imCol) >= static_cast<LongType>(iW) ||
                    static_cast<LongType>(imCol) < 0)
                  *col = zeroPadVal;
                else {
                  im = imBuff + b * imStride0 + c * imStride1 + imRow * imStride2 + imCol * imStride3;
                  *col = static_cast<T>(*im);
                }
              }
            }
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, bS, 1, 0, oH, 1);
}

void im2col(sd::LaunchContext& context, const NDArray& im, NDArray& col, const LongType kH, const LongType kW, const LongType sH,
            const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW, const NDArray& arrZeroPadVal) {
#if defined(HAVE_VEDA)
    NDArray::preparePrimaryUse({&col}, {&im});
#endif
  BUILD_SINGLE_SELECTOR(im.dataType(), im2col_, (context, im, col, kH, kW, sH, sW, pH, pW, dH, dW, arrZeroPadVal),
                        SD_FLOAT_TYPES);
#if defined(HAVE_VEDA)
    NDArray::registerPrimaryUse({&col}, {&im});
#endif
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif