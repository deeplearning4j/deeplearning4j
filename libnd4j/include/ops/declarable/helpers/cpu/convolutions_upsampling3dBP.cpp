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
static void upsampling3dBP_(const NDArray& gradO, NDArray& gradI, const bool isNCDHW) {
  // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
  // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW,
  // iC] (NDHWC)

  const T* x = gradO.bufferAsT<T>();
  T* z = gradI.bufferAsT<T>();

  const sd::LongType dimID = isNCDHW ? 2 : 1;
  const sd::LongType dimIC = isNCDHW ? 1 : 4;

  const sd::LongType bS = gradI.sizeAt(0);
  const sd::LongType iC = gradI.sizeAt(dimIC);
  const sd::LongType iD = gradI.sizeAt(dimID);
  const sd::LongType iH = gradI.sizeAt(dimID + 1);
  const sd::LongType iW = gradI.sizeAt(dimID + 2);

  const sd::LongType factorD = gradO.sizeAt(dimID) / iD;
  const sd::LongType factorH = gradO.sizeAt(dimID + 1) / iH;
  const sd::LongType factorW = gradO.sizeAt(dimID + 2) / iW;

  const sd::LongType xStride0 = gradO.stridesOf()[0];
  const sd::LongType xStride1 = gradO.stridesOf()[dimIC];
  const sd::LongType xStride2 = gradO.stridesOf()[dimID];
  const sd::LongType xStride3 = gradO.stridesOf()[dimID + 1];
  const sd::LongType xStride4 = gradO.stridesOf()[dimID + 2];

  const sd::LongType zStride0 = gradI.stridesOf()[0];
  const sd::LongType zStride1 = gradI.stridesOf()[dimIC];
  const sd::LongType zStride2 = gradI.stridesOf()[dimID];
  const sd::LongType zStride3 = gradI.stridesOf()[dimID + 1];
  const sd::LongType zStride4 = gradI.stridesOf()[dimID + 2];

  // loop through output array
  auto func = PRAGMA_THREADS_FOR_3D {
    for (sd::LongType b = start_x; b < stop_x; b += inc_x) {
      for (sd::LongType c = start_y; c < stop_y; c += inc_y) {
        for (sd::LongType d = start_z; d < stop_z; d += inc_z) {
          for (sd::LongType h = 0; h < iH; ++h) {
            for (sd::LongType w = 0; w < iW; ++w) {
              const auto zOffset = b * zStride0 + c * zStride1 + d * zStride2 + h * zStride3 + w * zStride4;

              z[zOffset] = 0;

              for (sd::LongType xd = d * factorD; xd < d * factorD + factorD; ++xd)
                for (sd::LongType xh = h * factorH; xh < h * factorH + factorH; ++xh)
                  for (sd::LongType xw = w * factorW; xw < w * factorW + factorW; ++xw)
                    z[zOffset] += x[b * xStride0 + c * xStride1 + xd * xStride2 + xh * xStride3 + xw * xStride4];
            }
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, iD, 1);
}

void ConvolutionUtils::upsampling3dBP(sd::graph::Context& block, const NDArray& gradO, NDArray& gradI,
                                      const bool isNCHW) {
  BUILD_SINGLE_SELECTOR(gradO.dataType(), upsampling3dBP_, (gradO, gradI, isNCHW), SD_FLOAT_TYPES);
}

}  // namespace ops
}  // namespace sd
