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
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/col2im.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {
namespace helpers {

// [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]
template <typename T>
void col2im_(sd::LaunchContext & context, const NDArray& input,  NDArray& output, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {

    auto imBuff         = output.bufferAsT<T>();
	auto colBuff        = input.bufferAsT<T>();
	auto imShapeBuffer  = output.getShapeInfo();
	auto colShapeBuffer = input.getShapeInfo();
    auto colShape  		= shape::shapeOf(colShapeBuffer);
    auto colStride 		= shape::stride(colShapeBuffer);
    auto imShape  	    = shape::shapeOf(imShapeBuffer);
    auto imStride 	    = shape::stride(imShapeBuffer);            

    const int bS = imShape[0];
    const int iC = imShape[1];
    const int kH = colShape[2];
    const int kW = colShape[3];                    
    const int oH = colShape[4];
    const int oW = colShape[5];
    const Nd4jLong colStride0 = colStride[0];
    const Nd4jLong colStride1 = colStride[1];
    const Nd4jLong colStride2 = colStride[2];
    const Nd4jLong colStride3 = colStride[3];
    const Nd4jLong colStride4 = colStride[4];
    const Nd4jLong colStride5 = colStride[5];
    const Nd4jLong imStride0  = imStride[0];
    const Nd4jLong imStride1  = imStride[1];
    const Nd4jLong imStride2  = imStride[2];
    const Nd4jLong imStride3  = imStride[3];
        
    memset(imBuff, 0, shape::length(imShapeBuffer) * sizeof(T));


    // if (shape::order(colShapeBuffer) == 'c' &&  shape::order(imShapeBuffer) == 'c' && shape::strideDescendingCAscendingF(colShapeBuffer) && shape::strideDescendingCAscendingF(imShapeBuffer)) {
    if (false) {

        auto func = PRAGMA_THREADS_FOR_2D {
            T *col, *im;
            int imRow, imCol;

            for (auto b = start_x; b < stop_x; b += inc_x) {
                for (auto c = start_y; c < stop_y; c += inc_y) {
                    for (int kRow = 0; kRow < kH; ++kRow) {
                        for (int kCol = 0; kCol < kW; ++kCol) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {

                                    imRow = (-pH + kRow * dH) + colH * sH;
                                    imCol = (-pW + kCol * dW) + colW * sW;

                                    col = colBuff + b * colStride0 + c * colStride1 + kRow * colStride2 + kCol * colStride3 + colH * colStride4 + colW * colStride5;
                                    im = imBuff + b * imStride0 + c * imStride1 + imRow * imStride2 + imCol * imStride3;

                                    if (static_cast<unsigned>(imRow) < static_cast<unsigned>(iH) &&
                                        static_cast<unsigned>(imCol) < static_cast<unsigned>(iW))
                                        *im += *col;
                                }
                            }
                        }
                    }
                }
            }
        };

        sd::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
    }
    else {

        auto func = PRAGMA_THREADS_FOR {
            T *col, *im;

            for (auto b = start; b < stop; b++) {
                T *im0 = imBuff + b * imStride0;
                T *col4 = colBuff + b * colStride0;
                for (int colH = 0; colH < oH; ++colH, col4 += colStride4) {
                    T *col5 = col4;
                    for (int colW = 0; colW < oW; ++colW, col5 += colStride5) {
                        T *col1 = col5;
                        T *im1 = im0;
                        for (int c = 0; c < iC; ++c, col1 += colStride1, im1 += imStride1) {
                            int imRow = (-pH + colH * sH);
                            T *col2 = col1;
                            T *im2 = im1 + imRow * imStride2;
                            for (int kRow = 0;
                                 kRow < kH; ++kRow, col2 += colStride2, imRow += dH, im2 += dH * imStride2) {
                                int imCol = -pW + colW * sW;
                                T *col3 = col2;
                                T *im3 = im2 + imCol * imStride3;
                                for (int kCol = 0;
                                     kCol < kW; ++kCol, col3 += colStride3, imCol += dW, im3 += dW * imStride3) {

                                    if (static_cast<unsigned>(imRow) < static_cast<unsigned>(iH) &&
                                        static_cast<unsigned>(imCol) < static_cast<unsigned>(iW))
                                        *im3 += *col3;
                                }
                            }
                        }
                    }
                }
            }
        };

        sd::Threads::parallel_tad(func, 0, bS);
    }
}


void col2im(sd::LaunchContext & context, const NDArray& input,  NDArray& output, const int sH, const int sW, const int pH, const int pW, const int iH, const int iW, const int dH, const int dW) {
	BUILD_SINGLE_SELECTOR(input.dataType(), col2im_, (context, input, output, sH, sW, pH, pW, iH, iW, dH, dW), FLOAT_TYPES);
}

}
}
}
