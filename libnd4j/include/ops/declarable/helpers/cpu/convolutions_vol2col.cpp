/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <ops/declarable/helpers/convolutions.h>
#include <execution/Threads.h>

namespace sd {
    namespace ops  {


//////////////////////////////////////////////////////////////////////////
// [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]
template <typename T>
static void vol2col_(const NDArray& volume, NDArray& columns, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

            const int bS = volume.sizeAt(0);
            const int iC = volume.sizeAt(1);
            const int iD = volume.sizeAt(2);
            const int iH = volume.sizeAt(3);
            const int iW = volume.sizeAt(4);
            const int kD = columns.sizeAt(2);
            const int kH = columns.sizeAt(3);
            const int kW = columns.sizeAt(4);
            const int oD = columns.sizeAt(5);
            const int oH = columns.sizeAt(6);
            const int oW = columns.sizeAt(7);
            const Nd4jLong colStride0 = columns.stridesOf()[0];
            const Nd4jLong colStride1 = columns.stridesOf()[1];
            const Nd4jLong colStride2 = columns.stridesOf()[2];
            const Nd4jLong colStride3 = columns.stridesOf()[3];
            const Nd4jLong colStride4 = columns.stridesOf()[4];
            const Nd4jLong colStride5 = columns.stridesOf()[5];
            const Nd4jLong colStride6 = columns.stridesOf()[6];
            const Nd4jLong colStride7 = columns.stridesOf()[7];
            const Nd4jLong volStride0 = volume.stridesOf()[0];
            const Nd4jLong volStride1 = volume.stridesOf()[1];
            const Nd4jLong volStride2 = volume.stridesOf()[2];
            const Nd4jLong volStride3 = volume.stridesOf()[3];
            const Nd4jLong volStride4 = volume.stridesOf()[4];

            T* colBuff = columns.bufferAsT<T>();
            T* volBuff = const_cast<NDArray&>(volume).bufferAsT<T>();


            if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo())) {

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

                                                    col = colBuff + b * colStride0 + c * colStride1 + kDep * colStride2 + kRow * colStride3 + kCol * colStride4 + colD * colStride5 + colH * colStride6 + colW * colStride7;

                                                    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
                                                        *col = static_cast<T>(0.);
                                                    else {
                                                        vol = volBuff + b * volStride0 + c * volStride1 + volDep * volStride2 + volRow * volStride3 + volCol * volStride4;
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
                    for (int b = start_x; b < stop_x; b++) {
                        for (int colD = start_y; colD < stop_y; colD++) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {
                                    for (int c = 0; c < iC; ++c) {
                                        for (int kDep = 0; kDep < kD; ++kDep) {
                                            for (int kRow = 0; kRow < kH; ++kRow) {
                                                for (int kCol = 0; kCol < kW; ++kCol) {

                                                    volDep = (-pD + kDep * dD) + colD * sD;
                                                    volRow = (-pH + kRow * dH) + colH * sH;
                                                    volCol = (-pW + kCol * dW) + colW * sW;

                                                    col = colBuff + b * colStride0 + c * colStride1 + kDep * colStride2 + kRow * colStride3 + kCol * colStride4 + colD * colStride5 + colH * colStride6 + colW * colStride7;

                                                    if (static_cast<unsigned>(volDep) >= static_cast<unsigned>(iD) || static_cast<unsigned>(volRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(volCol) >= static_cast<unsigned>(iW))
                                                        *col = static_cast<T>(0.f);
                                                    else {
                                                        vol = volBuff + b * volStride0 + c * volStride1 + volDep * volStride2 + volRow * volStride3 + volCol * volStride4;
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
                //func(0, 0, bS, 1, 0, oD, 1);
            }
        }

void ConvolutionUtils::vol2col(sd::graph::Context& block, const NDArray& volume, NDArray& columns, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
     BUILD_SINGLE_SELECTOR(volume.dataType(), vol2col_, (volume, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);
}

}
}