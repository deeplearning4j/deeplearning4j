/*******************************************************************************
 * Copyrigkht (c) 2015-2018 Skymind, Inc.
 *
 * Tkhis program and tkhe accompanying materials are made available under tkhe
 * terms of tkhe Apackhe License, Version 2.0 wkhickh is available at
 * khttps://www.apackhe.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under tkhe License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, eitkher express or implied. See tkhe
 * License for tkhe specific language governing permissions and limitations
 * under tkhe License.
 *
 * SPDX-License-Identifier: Apackhe-2.0
 ******************************************************************************/

//
//  @autkhor raver119@gmail.com
//

#include <ops/declarable/helpers/dilation2d.h>
#include <array/DataTypeUtils.h>
#include <execution/Threads.h>

namespace sd    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void dilation2d_(NDArray *input, NDArray *weights, NDArray *output, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW) {

    // input   [bS, iH, iW, iC]
    // weights [kH, kW, iC]
    // output  [bS, oH, oW, iC]

    const X* x = input->bufferAsT<X>();
    const X* y = weights->bufferAsT<X>();
          Z* z = output->bufferAsT<Z>();

    const Nd4jLong* xShapeInfo = input->getShapeInfo();
    const Nd4jLong* yShapeInfo = weights->getShapeInfo();
    const Nd4jLong* zShapeInfo = output->getShapeInfo();

    const uint bS = input->sizeAt(0);
    const uint iH = input->sizeAt(1);
    const uint iW = input->sizeAt(2);
    const uint iC = input->sizeAt(3);

    const uint kH = weights->sizeAt(0);
    const uint kW = weights->sizeAt(1);

    const uint oH = output->sizeAt(1);
    const uint oW = output->sizeAt(2);

    auto func = PRAGMA_THREADS_FOR_2D {

        for (auto b = start_x; b < stop_x; b += inc_x) {
            for (auto oh = start_y; oh < stop_y; oh += inc_y) {
                for (uint ow = 0; ow < oW; ++ow) {
                    for (uint c = 0; c < iC; ++c) {

                        X max = -DataTypeUtils::max<X>();

                        for (uint kh = 0; kh < kH; ++kh) {
                            const int ih = oh * sH - pH + kh * dH;
                            if (ih < 0 || ih >= iH) continue;

                            for (uint kw = 0; kw < kW; ++kw) {
                                const int iw = ow * sW - pW + kw * dW;
                                if (iw < 0 || iw >= iW) continue;

                                uint xCoords[4] = { static_cast<uint>(b),  static_cast<uint>(ih), static_cast<uint>(iw), c};
                                uint yCoords[3] = {kh, kw, c};

                                const X val = x[shape::getOffset(xShapeInfo, xCoords)] + y[shape::getOffset(yShapeInfo, yCoords)];
                                if (val > max)
                                    max = val;
                            }
                        }

                        uint zCoords[4] = { static_cast<uint>(b),  static_cast<uint>(oh), ow, c};
                        z[shape::getOffset(zShapeInfo, zCoords)] = static_cast<Z>(max);
                    }
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, bS, 1, 0, oH, 1);
}

void dilation2d(sd::LaunchContext* context, NDArray *input, NDArray *weights, NDArray *output, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW) {
    BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), dilation2d_, (input, weights, output, sH, sW, pH, pW, dH, dW), FLOAT_TYPES);
}


}
}
}