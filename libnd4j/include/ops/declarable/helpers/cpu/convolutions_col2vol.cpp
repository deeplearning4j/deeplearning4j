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
// [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to [bS, iC, iD, iH, iW]
template <typename T>
static void col2vol_(const NDArray& columns, NDArray& volume, const LongType sD, const LongType sH, const LongType sW, const LongType pD,
                     const LongType pH, const LongType pW, const LongType dD, const LongType dH, const LongType dW) {
  // initial zeroing of volume content
  volume.nullify();

  const LongType bS = volume.sizeAt(0);
  const LongType iC = volume.sizeAt(1);
  const LongType iD = volume.sizeAt(2);
  const LongType iH = volume.sizeAt(3);
  const LongType iW = volume.sizeAt(4);
  const LongType kD = columns.sizeAt(2);
  const LongType kH = columns.sizeAt(3);
  const LongType kW = columns.sizeAt(4);
  const LongType oD = columns.sizeAt(5);
  const LongType oH = columns.sizeAt(6);
  const LongType oW = columns.sizeAt(7);
  const sd::LongType colStride0 = columns.stridesOf()[0];
  const sd::LongType colStride1 = columns.stridesOf()[1];
  const sd::LongType colStride2 = columns.stridesOf()[2];
  const sd::LongType colStride3 = columns.stridesOf()[3];
  const sd::LongType colStride4 = columns.stridesOf()[4];
  const sd::LongType colStride5 = columns.stridesOf()[5];
  const sd::LongType colStride6 = columns.stridesOf()[6];
  const sd::LongType colStride7 = columns.stridesOf()[7];
  const sd::LongType volStride0 = volume.stridesOf()[0];
  const sd::LongType volStride1 = volume.stridesOf()[1];
  const sd::LongType volStride2 = volume.stridesOf()[2];
  const sd::LongType volStride3 = volume.stridesOf()[3];
  const sd::LongType volStride4 = volume.stridesOf()[4];

  T* volBuff = volume.bufferAsT<T>();
  T* colBuff = const_cast<NDArray&>(columns).bufferAsT<T>();

  if (volume.ordering() == 'c' && columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.shapeInfo()) &&
      shape::strideDescendingCAscendingF(columns.shapeInfo())) {
    auto func = PRAGMA_THREADS_FOR {
      T *col, *vol;
      sd::LongType volDep, volRow, volCol;

      for (sd::LongType b = start; b < stop; b++) {
        for (sd::LongType c = 0; c < iC; c++) {
          for (sd::LongType kDep = 0; kDep < kD; ++kDep) {
            for (sd::LongType kRow = 0; kRow < kH; ++kRow) {
              for (sd::LongType kCol = 0; kCol < kW; ++kCol) {
                for (sd::LongType colD = 0; colD < oD; ++colD) {
                  for (sd::LongType colH = 0; colH < oH; ++colH) {
                    for (sd::LongType colW = 0; colW < oW; ++colW) {
                      volDep = (-pD + kDep * dD) + colD * sD;
                      volRow = (-pH + kRow * dH) + colH * sH;
                      volCol = (-pW + kCol * dW) + colW * sW;

                      if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) &&
                          static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) &&
                          static_cast<unsigned>(volCol) < static_cast<unsigned>(iW)) {

                        auto colIndex = b * colStride0 + c * colStride1 + kDep * colStride2 + kRow * colStride3 +
                                        kCol * colStride4 + colD * colStride5 + colH * colStride6 + colW * colStride7;
                        auto volIndex = b * volStride0 + c * volStride1 + volDep * volStride2 + volRow * volStride3 +
                                        volCol * volStride4;



                        col = colBuff + colIndex;
                        vol = volBuff + volIndex;
                        *vol += *col;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    };

    samediff::Threads::parallel_tad(func, 0, bS);

  } else {
    auto func = PRAGMA_THREADS_FOR {
      T *col, *vol;
      sd::LongType volDep, volRow, volCol;

      for (sd::LongType b = start; b < stop; b++) {
        for (sd::LongType colD = 0; colD < oD; colD++) {
          for (sd::LongType colH = 0; colH < oH; ++colH) {
            for (sd::LongType colW = 0; colW < oW; ++colW) {
              for (sd::LongType c = 0; c < iC; ++c) {
                for (sd::LongType kDep = 0; kDep < kD; ++kDep) {
                  for (sd::LongType kRow = 0; kRow < kH; ++kRow) {
                    for (sd::LongType kCol = 0; kCol < kW; ++kCol) {
                      volDep = (-pD + kDep * dD) + colD * sD;
                      volRow = (-pH + kRow * dH) + colH * sH;
                      volCol = (-pW + kCol * dW) + colW * sW;

                      if (volDep >= 0 && volDep < iD &&
                          volRow >= 0 && volRow < iH &&
                          volCol >= 0 && volCol < iW) {

                        auto colIndex = b * colStride0 + c * colStride1 + kDep * colStride2 + kRow * colStride3 +
                                        kCol * colStride4 + colD * colStride5 + colH * colStride6 + colW * colStride7;
                        auto volIndex = b * volStride0 + c * volStride1 + volDep * volStride2 + volRow * volStride3 +
                                        volCol * volStride4;

                        col = colBuff + colIndex;
                        vol = volBuff + volIndex;
                        *vol += *col;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    };

    samediff::Threads::parallel_tad(func, 0, bS);
  }
}

void ConvolutionUtils::col2vol(sd::graph::Context& block, const NDArray& columns, NDArray& volume, const LongType sD,
                               const LongType sH, const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD,
                               const LongType dH, const LongType dW) {
  BUILD_SINGLE_SELECTOR(volume.dataType(), col2vol_, (columns, volume, sD, sH, sW, pD, pH, pW, dD, dH, dW),
                        SD_FLOAT_TYPES);
}

}  // namespace ops
}  // namespace sd
