/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @autkhor raver119@gmail.com
//
#include <array/DataTypeUtils.h>
#include <execution/Threads.h>
#include <ops/declarable/helpers/dilation2d.h>
#if NOT_EXCLUDED(OP_dilation2d)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void dilation2d_(NDArray* input, NDArray* weights, NDArray* output, const sd::LongType sH, const sd::LongType sW, const sd::LongType pH,
                        const sd::LongType pW, const sd::LongType dH, const sd::LongType dW) {
  // input   [bS, iH, iW, iC]
  // weights [kH, kW, iC]
  // output  [bS, oH, oW, iC]

  const X* x = input->bufferAsT<X>();
  const X* y = weights->bufferAsT<X>();
  Z* z = output->bufferAsT<Z>();

  const sd::LongType* xShapeInfo = input->shapeInfo();
  const sd::LongType* yShapeInfo = weights->shapeInfo();
  const sd::LongType* zShapeInfo = output->shapeInfo();

  const sd::LongType bS = input->sizeAt(0);
  const sd::LongType iH = input->sizeAt(1);
  const sd::LongType iW = input->sizeAt(2);
  const sd::LongType iC = input->sizeAt(3);

  const sd::LongType kH = weights->sizeAt(0);
  const sd::LongType kW = weights->sizeAt(1);

  const sd::LongType oH = output->sizeAt(1);
  const sd::LongType oW = output->sizeAt(2);

  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto b = start_x; b < stop_x; b += inc_x) {
      for (auto oh = start_y; oh < stop_y; oh += inc_y) {
        for (sd::LongType ow = 0; ow < oW; ++ow) {
          for (sd::LongType c = 0; c < iC; ++c) {
            X max = -DataTypeUtils::max<X>();

            for (sd::LongType kh = 0; kh < kH; ++kh) {
              const int ih = oh * sH - pH + kh * dH;
              if (ih < 0 || ih >= iH) continue;

              for (sd::LongType kw = 0; kw < kW; ++kw) {
                const int iw = ow * sW - pW + kw * dW;
                if (iw < 0 || iw >= iW) continue;

                sd::LongType  xCoords[4] = {static_cast<sd::LongType >(b), static_cast<sd::LongType >(ih),
                                           static_cast<sd::LongType >(iw), c};
                sd::LongType  yCoords[3] = {kh, kw, c};

                const X val = x[shape::getOffset(xShapeInfo, xCoords)] + y[shape::getOffset(yShapeInfo, yCoords)];
                if (val > max) max = val;
              }
            }

            sd::LongType  zCoords[4] = {static_cast<sd::LongType >(b), static_cast<sd::LongType >(oh), ow, c};
            z[shape::getOffset(zShapeInfo, zCoords)] = static_cast<Z>(max);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, bS, 1, 0, oH, 1);
}

void dilation2d(sd::LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, const sd::LongType sH,
                const sd::LongType sW, const sd::LongType pH, const sd::LongType pW, const sd::LongType dH, const sd::LongType dW) {
  BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), dilation2d_, (input, weights, output, sH, sW, pH, pW, dH, dW),
                              SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif