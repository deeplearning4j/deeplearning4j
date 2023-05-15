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
// [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]
template <typename T>
static void vol2col_(const NDArray& volume, NDArray& columns, const LongType sD, const LongType sH, const LongType sW, const LongType pD,
                     const LongType pH, const LongType pW, const LongType dD, const LongType dH, const LongType dW) {
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

  T* colBuff = columns.bufferAsT<T>();
  T* volBuff = const_cast<NDArray&>(volume).bufferAsT<T>();

  if (volume.ordering() == 'c' && columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.shapeInfo()) &&
      shape::strideDescendingCAscendingF(columns.shapeInfo())) {
    auto func = PRAGMA_THREADS_FOR_3D {
      T *col, *vol;
      int volDep, volRow, volCol;

      for (int b = start_x; b < stop_x; b += inc_x) {
        for (int c = start_y; c < stop_y; c += inc_y) {
          for (int kDep = start_z; kDep < stop_z; kDep += inc_z) {
            for (int kRow = 0; kRow < kH; ++kRow) {
              for (int kCol = 0; kCol < kW; ++kCol) {
                for (int colD = 0; colD < oD; ++colD) {
                  for (int colH = 0; colH < oH; ++colH) {
                    for (int colW = 0; colW < oW; ++colW) {
                      volDep = (-pD + kDep * dD) + colD * sD;
                      volRow = (-pH + kRow * dH) + colH * sH;
                      volCol = (-pW + kCol * dW) + colW * sW;

                      col = colBuff + b * colStride0 + c * colStride1 + kDep * colStride2 + kRow * colStride3 +
                            kCol * colStride4 + colD * colStride5 + colH * colStride6 + colW * colStride7;

                      if (static_cast<LongType>(volDep) >= static_cast<LongType>(iD) ||
                          static_cast<LongType>(volRow) >= static_cast<LongType>(iH) ||
                          static_cast<LongType>(volCol) >= static_cast<LongType>(iW))
                        *col = static_cast<T>(0.);
                      else {
                        vol = volBuff + b * volStride0 + c * volStride1 + volDep * volStride2 + volRow * volStride3 +
                              volCol * volStride4;
                        *col = *vol;
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

    samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, kD, 1);

  } else {
    auto func = PRAGMA_THREADS_FOR_2D {
      T *col, *vol;
      int volDep, volRow, volCol;
      for (LongType b = start_x; b < stop_x; b++) {
        for (LongType colD = start_y; colD < stop_y; colD++) {
          for (LongType colH = 0; colH < oH; ++colH) {
            for (LongType colW = 0; colW < oW; ++colW) {
              for (LongType c = 0; c < iC; ++c) {
                for (LongType kDep = 0; kDep < kD; ++kDep) {
                  for (LongType kRow = 0; kRow < kH; ++kRow) {
                    for (LongType kCol = 0; kCol < kW; ++kCol) {
                      volDep = (-pD + kDep * dD) + colD * sD;
                      volRow = (-pH + kRow * dH) + colH * sH;
                      volCol = (-pW + kCol * dW) + colW * sW;

                      col = colBuff + b * colStride0 + c * colStride1 + kDep * colStride2 + kRow * colStride3 +
                            kCol * colStride4 + colD * colStride5 + colH * colStride6 + colW * colStride7;

                      if (static_cast<LongType>(volDep) >= static_cast<LongType>(iD) ||
                          static_cast<LongType>(volRow) >= static_cast<LongType>(iH) ||
                          static_cast<LongType>(volCol) >= static_cast<LongType>(iW))
                        *col = static_cast<T>(0.f);
                      else {
                        vol = volBuff + b * volStride0 + c * volStride1 + volDep * volStride2 + volRow * volStride3 +
                              volCol * volStride4;
                        *col = *vol;
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

    samediff::Threads::parallel_for(func, 0, bS, 1, 0, oD, 1);
    // func(0, 0, bS, 1, 0, oD, 1);
  }
}

void ConvolutionUtils::vol2col(sd::graph::Context& block, const NDArray& volume, NDArray& columns, const LongType sD,
                               const LongType sH, const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD,
                               const LongType dH, const LongType dW) {
  BUILD_SINGLE_SELECTOR(volume.dataType(), vol2col_, (volume, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW),
                        SD_FLOAT_TYPES);
}

}  // namespace ops
}  // namespace sd
