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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.09.2018
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/convolutions.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void upsampling2d_(const NDArray& input, NDArray& output, const LongType factorH, const LongType factorW,
                          const bool isNCHW) {
  // input  has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)
  // output has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)

  const T* x = input.bufferAsT<T>();
  T* z = output.bufferAsT<T>();

  const sd::LongType dimIH = isNCHW ? 2 : 1;
  const sd::LongType dimIC = isNCHW ? 1 : 3;

  const sd::LongType bS = input.sizeAt(0);
  const sd::LongType iC = input.sizeAt(dimIC);
  const sd::LongType oH = output.sizeAt(dimIH);
  const sd::LongType oW = output.sizeAt(dimIH + 1);

  const sd::LongType xStride0 = input.stridesOf()[0];
  const sd::LongType xStride1 = input.stridesOf()[dimIC];
  const sd::LongType xStride2 = input.stridesOf()[dimIH];
  const sd::LongType xStride3 = input.stridesOf()[dimIH + 1];

  const sd::LongType zStride0 = output.stridesOf()[0];
  const sd::LongType zStride1 = output.stridesOf()[dimIC];
  const sd::LongType zStride2 = output.stridesOf()[dimIH];
  const sd::LongType zStride3 = output.stridesOf()[dimIH + 1];

  // loop through output array
  auto func = PRAGMA_THREADS_FOR_3D {
    sd::Unsigned xCoord2, xCoord3;
    for (sd::Unsigned b = start_x; b < stop_x; b += inc_x) {
      for (sd::Unsigned c = start_y; c < stop_y; c += inc_y) {
        for (sd::Unsigned h = start_z; h < stop_z; h += inc_z) {
          for (sd::Unsigned w = 0; w < oW; ++w) {
            xCoord2 = h / factorH;
            xCoord3 = w / factorW;

            z[b * zStride0 + c * zStride1 + h * zStride2 + w * zStride3] =
                x[b * xStride0 + c * xStride1 + xCoord2 * xStride2 + xCoord3 * xStride3];
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oH, 1);
}

void ConvolutionUtils::upsampling2d(sd::graph::Context& block, const NDArray& input, NDArray& output, const LongType factorH,
                                    const LongType factorW, const bool isNCHW) {
  BUILD_SINGLE_SELECTOR(input.dataType(), upsampling2d_, (input, output, factorH, factorW, isNCHW), SD_FLOAT_TYPES);
}

}  // namespace ops
}  // namespace sd
