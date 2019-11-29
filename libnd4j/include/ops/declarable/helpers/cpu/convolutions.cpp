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
#include<ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/col2im.h>
#include <NDArrayFactory.h>
#include <MmulHelper.h>
#include <execution/Threads.h>

namespace nd4j {
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

//////////////////////////////////////////////////////////////////////////
// [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to [bS, iC, iD, iH, iW]
        template <typename T>
        static void col2vol_(const NDArray& columns, NDArray& volume, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

            // initial zeroing of volume content
            volume.nullify();

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

            T* volBuff = volume.bufferAsT<T>();
            T* colBuff = const_cast<NDArray&>(columns).bufferAsT<T>();


            if (volume.ordering() == 'c' &&  columns.ordering() == 'c' && shape::strideDescendingCAscendingF(volume.getShapeInfo()) && shape::strideDescendingCAscendingF(columns.getShapeInfo())) {

                auto func = PRAGMA_THREADS_FOR {
                    T* col, *vol;
                    int volDep, volRow, volCol;

                    for (int b = start; b < stop; b++) {
                        for (int c = 0; c < iC; c++) {
                            for (int kDep = 0; kDep < kD; ++kDep) {
                                for (int kRow = 0; kRow < kH; ++kRow) {
                                    for (int kCol = 0; kCol < kW; ++kCol) {
                                        for (int colD = 0; colD < oD; ++colD) {
                                            for (int colH = 0; colH < oH; ++colH) {
                                                for (int colW = 0; colW < oW; ++colW) {

                                                    volDep = -pD + kDep * dD + colD * sD;
                                                    volRow = -pH + kRow * dH + colH * sH;
                                                    volCol = -pW + kCol * dW + colW * sW;

                                                    if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) && static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(volCol) < static_cast<unsigned>(iW)) {
                                                        col = colBuff + b * colStride0 + c * colStride1 + kDep * colStride2 + kRow * colStride3 + kCol * colStride4 + colD * colStride5 + colH * colStride6 + colW * colStride7;
                                                        vol = volBuff + b * volStride0 + c * volStride1 + volDep * volStride2 + volRow * volStride3 + volCol * volStride4;
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
                    T* col, *vol;
                    int volDep, volRow, volCol;

                    for (int b = start; b < stop; b++) {
                        for (int colD = 0; colD < oD; colD++) {
                            for (int colH = 0; colH < oH; ++colH) {
                                for (int colW = 0; colW < oW; ++colW) {
                                    for (int c = 0; c < iC; ++c) {
                                        for (int kDep = 0; kDep < kD; ++kDep) {
                                            for (int kRow = 0; kRow < kH; ++kRow) {
                                                for (int kCol = 0; kCol < kW; ++kCol) {

                                                    volDep = (-pD + kDep * dD) + colD * sD;
                                                    volRow = (-pH + kRow * dH) + colH * sH;
                                                    volCol = (-pW + kCol * dW) + colW * sW;

                                                    if (static_cast<unsigned>(volDep) < static_cast<unsigned>(iD) && static_cast<unsigned>(volRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(volCol) < static_cast<unsigned>(iW)) {
                                                        col = colBuff + b * colStride0 + c * colStride1 + kDep * colStride2 + kRow * colStride3 + kCol * colStride4 + colD * colStride5 + colH * colStride6 + colW * colStride7;
                                                        vol = volBuff + b * volStride0 + c * volStride1 + volDep * volStride2 + volRow * volStride3 + volCol * volStride4;
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


//////////////////////////////////////////////////////////////////////////
        template <typename X, typename Y>
        static void conv2d_(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {

            // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
            // weights [kH, kW, iC, oC] always
            // bias    [oC]
            // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

            // kH  filter(kernel) height
            // kW  filter(kernel) width
            // sH  strides height
            // sW  strides width
            // pH  paddings height
            // pW  paddings width
            // dH  dilations height
            // dW  dilations width
            // paddingMode 0-VALID, 1-SAME
            // isNCHW      1-NCHW,  0-NHWC

            int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
            int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

            ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

            nd4j_debug("MKL-DNN is not used for conv2d!\n", 0);

            std::vector<int> permutForOutput;

            if(isNCHW)
                permutForOutput = {0, 3, 1, 2};                                             // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
            else
                input = new NDArray(input->permute({0, 3, 1, 2}));                         // [bS, iH, iW, iC] -> [bS, iC, iH, iW] if NHWC

            NDArray col('c', {bS, oH, oW, kH, kW, iC}, input->dataType(), input->getContext());
            NDArray colP = col.permute({0, 5, 3, 4, 1, 2});            // {bS, iC, kH, kW, oH, oW}
            NDArray mmulResult('f', {bS*oH*oW, oC}, output->dataType(), output->getContext());

            //----- calculation of output -----//
            auto ctx = block.launchContext();
            helpers::im2col(*ctx, *input, colP, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
            MmulHelper::tensorDot(&col, weights, &mmulResult, {3,4,5}, {0,1,2}, {}); // [bS, oH, oW, kH, kW, iC] x [kH, kW, iC, oC] = [bS, oH, oW, oC]

            //----- assign outTemp to output  -----//
            if(isNCHW) {
                mmulResult.reshapei({bS, oH, oW, oC});
                mmulResult.permutei(permutForOutput);
            }
            output->assign(mmulResult);

            //----- add biases if required -----//
            if(bias)
                // output->applyBroadcast(broadcast::Add, {indIOioC}, bias);
                helpers::addBias(block, *output, *bias, *output, isNCHW);

            if(!isNCHW)
                delete input;

        }

//////////////////////////////////////////////////////////////////////////
        template <typename X, typename Y>
        static void conv2dBP_(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {

            // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
            // weights [kH, kW, iC, oC] always
            // bias    [oC]
            // gradO   [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

            // gradI    [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
            // gradW    [kH, kW, iC, oC] always
            // gradB    [oC]

            // kH         filter(kernel) height
            // kW         filter(kernel) width
            // sH         strides height
            // sW         strides width
            // pH         paddings height
            // pW         paddings width
            // dH         dilations height
            // dW         dilations width
            // paddingMode 0-VALID, 1-SAME
            // isNCHW      0-NHWC, 1-NCHW

            int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
            int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

            ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

            nd4j_debug("MKL-DNN is not used for conv2d_bp!\n", 0);

            std::vector<int> gradOaxesForDot;

            if(!isNCHW) {
                gradOaxesForDot  = {0, 1, 2};                                           // bS, oH, oW
                input = new NDArray(input->permute({0, 3, 1, 2}));                      // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
                gradI = new NDArray(gradI->permute({0, 3, 1, 2}));                      // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
            } else {
                gradOaxesForDot  = {0, 2, 3};                                           // bS, oH, oW
            }

            NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());

            // ----- calculation of gradW ----- //
            if(gradW) {
                auto ctx = block.launchContext();
                helpers::im2col(*ctx, *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));   // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
                nd4j::MmulHelper::tensorDot(&columns, gradO, gradW, {0,4,5}, gradOaxesForDot, {2, 0, 1, 3});       // [bS, iC, kH, kW, oH, oW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, oC]
            }

            // ----- calculation of gradB ----- //
            if(gradB) {
                NDArray* gradBR = gradB;
                if(gradB->rankOf() == 2)
                    gradBR = new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()}));
                gradO->reduceAlongDimension(reduce::Sum, gradBR, gradOaxesForDot);                          // sum over bS, oH, oW
                if(gradBR != gradB)
                    delete gradBR;
            }

            //----- calculation of gradI -----//
            nd4j::MmulHelper::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, {2, 3, 1, 0, 4, 5});  // [kH, kW, iC, oC]/[oC, iC, kH, kW]] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]

            helpers::col2im(*block.launchContext(), columns, *gradI, sH, sW, pH, pW, iH, iW, dH, dW);                          // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

            if(!isNCHW) {
                delete input;
                delete gradI;
            }
        }

//////////////////////////////////////////////////////////////////////////
        template <typename X, typename Y>
        static void depthwiseConv2d_(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {

            // input     [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
            // weights   [kH, kW, iC, mC] always
            // bias      [oC] = iC*mC
            // output    [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

            // kH           filter(kernel) height
            // kW           filter(kernel) width
            // sH           strides height
            // sW           strides width
            // pH           paddings height
            // pW           paddings width
            // dH           dilations height
            // dW           dilations width
            // paddingMode  0-VALID, 1-SAME
            // isNCHW       0-NCHW,  1-NHWC

            int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
            int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
            mC = weights->sizeAt(indWmC);                           // channels multiplier

            std::vector<std::vector<Nd4jLong>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
            std::vector<std::vector<Nd4jLong>> modifOutput;
            std::vector<Nd4jLong> outReShape;

            if(!isNCHW) {
                outReShape = {bS, oH, oW, iC, mC};                                              // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
                modifOutput = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
                input = new NDArray(input->permute({0, 3, 1, 2}));                             // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
            }
            else {
                outReShape = {bS, iC, mC, oH, oW};                                              // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
                modifOutput = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
            }

            if(paddingMode == 1)                       // SAME
                ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

            NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());
            NDArray outputReshaped = output->reshape(output->ordering(), outReShape);

            helpers::im2col(*output->getContext(), *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
            MmulHelper::tensorDot(&columns, weights, &outputReshaped, modifColumns, {{2,0,1,3},{iC,kH*kW,mC}}, modifOutput);              // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]

            if(bias)
                // output->applyBroadcast(broadcast::Add, {indIOioC}, bias);
                helpers::addBias(block, *output, *bias, *output, isNCHW);

            if(!isNCHW)
                delete input;
        }

//////////////////////////////////////////////////////////////////////////
        template <typename X, typename Y>
        static void depthwiseConv2dBP_(const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {

            // input    [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
            // weights  [kH, kW, iC, mC] always
            // bias     [oC] = [iC*mC]
            // gradO    [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
            // gradI    [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
            // gradW    [kH, kW, iC, mC] always
            // gradB    [oC]

            //  kH          filter(kernel) height
            //  kW          filter(kernel) width
            //  sH          strides height
            //  sW          strides width
            //  pH          paddings height
            //  pW          paddings width
            //  dH          dilations height
            //  dW          dilations width
            //  paddingMode 0-VALID, 1-SAME
            //  isNCHW      0-NHWC, 1-NCHW

            int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
            int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
            mC = weights->sizeAt(indWmC);                           // channels multiplier

            std::vector<std::vector<Nd4jLong>> modifColumns = {{1,2,3,0,4,5}, {iC, kH*kW, bS*oH*oW}};      // [bS,iC,kH,kW,oH,oW] -> [iC, kH*kW, bS*oH*oW]
            std::vector<std::vector<Nd4jLong>> modifGradO1, modifGradO2;
            std::vector<Nd4jLong> gradOreShape;

            if(!isNCHW) {
                gradOreShape = {bS, oH, oW, iC, mC};                                            // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
                modifGradO1 = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
                modifGradO2 = {{3,0,1,2},{iC, mC, bS*oH*oW}};                                   // [bS,oH,oW,iC*mC] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
                input = new NDArray(input->permute({0, 3, 1, 2}));                             // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
                gradI = new NDArray(gradI->permute({0, 3, 1, 2}));                             // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
            }
            else {
                gradOreShape = {bS, iC, mC, oH, oW};                                            // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
                modifGradO1 = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
                modifGradO2 = {{1,0,2,3},{iC, mC, bS*oH*oW}};                                   // [bS,iC*mC,oH,oW] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
            }

            if(paddingMode == 1)                       // SAME
                ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

            NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());
            NDArray gradOreshaped = gradO->reshape(gradO->ordering(), gradOreShape);

            // ----- calculation of gradW and gradB ----- //

            helpers::im2col(*input->getContext(), *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
            nd4j::MmulHelper::tensorDot(&columns, &gradOreshaped, gradW, modifColumns, modifGradO1, {{2,0,1,3},{iC,kH*kW,mC}});  // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

            // ----- calculation of gradB ----- //
            if(gradB) {
                NDArray* gradBR = gradB;
                if(gradB->rankOf() == 2)
                    gradBR = new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()}));
                gradO->reduceAlongDimension(reduce::Sum, gradBR, {0,indOoH,indOoH+1});                      // sum over bS, oH, oW

                if(gradBR != gradB)
                    delete gradBR;
            }

            //----- calculation of gradI -----//
            nd4j::MmulHelper::tensorDot(weights, gradO, &columns, {{2,0,1,3},{iC,kH*kW,mC}}, modifGradO2, modifColumns); // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]
            helpers::col2im(*input->getContext(), columns, *gradI, sH, sW, pH, pW, iH, iW, dH, dW);                                       // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

            if(!isNCHW) {
                delete input;
                delete gradI;
            }
        }

//////////////////////////////////////////////////////////////////////////
        template <typename X, typename Y>
        static void sconv2d_(nd4j::graph::Context& block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {

            // input         [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
            // weightsDepth  [kH, kW, iC, mC]  always
            // weightsPoint  [1, 1, iC*mC, oC] always
            // bias          [oC], oC = iC*mC if weightsPoint=nullptr
            // output is     [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW]  (NCHW)

            //  kH         filter(kernel) height
            //  kW         filter(kernel) width
            //  sH         strides height
            //  sW         strides width
            //  pH         paddings height
            //  pW         paddings width
            //  dH         dilations height
            //  dW         dilations width
            //  paddingMode 0-VALID, 1-SAME
            //  isNCHW      1-NCHW,  0-NHWC

            int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
            int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
            mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

            NDArray* outputDepth = output;
            if(weightsPoint)                        // if pointwise convolution is expected
                outputDepth = new NDArray(output->ordering(), !isNCHW ? std::vector<Nd4jLong>({bS, oH, oW, iC*mC}) : std::vector<Nd4jLong>({bS, iC*mC, oH, oW}), input->dataType(), input->getContext());

            // ----- perform depthwise convolution (if weightsPoint is absent then oC = iC*mC) ----- //
            ConvolutionUtils::depthwiseConv2d(block, input, weightsDepth, weightsPoint ? nullptr : bias, outputDepth, kH,kW, sH,sW, pH,pW, dH,dW, paddingMode, isNCHW);

            // ----- perform pointwise convolution (oH = iH, oW = iW) ----- //
            if (weightsPoint) {
                ConvolutionUtils::conv2d(block, outputDepth, weightsPoint, bias, output, 1,1, 1,1, 0,0, 1,1, paddingMode, isNCHW);             // in this case oH=iH, oW=iW
                delete outputDepth;
            }
        }

//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void upsampling2d_(const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW) {
            // input  has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)
            // output has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)

            const T* x = input.bufferAsT<T>();
                  T* z = output.bufferAsT<T>();

            const uint dimIH = isNCHW ? 2 : 1;
            const uint dimIC = isNCHW ? 1 : 3;

            const uint bS = input.sizeAt(0);
            const uint iC = input.sizeAt(dimIC);
            const uint oH = output.sizeAt(dimIH);
            const uint oW = output.sizeAt(dimIH + 1);

            const Nd4jLong xStride0 = input.stridesOf()[0];
            const Nd4jLong xStride1 = input.stridesOf()[dimIC];
            const Nd4jLong xStride2 = input.stridesOf()[dimIH];
            const Nd4jLong xStride3 = input.stridesOf()[dimIH + 1];

            const Nd4jLong zStride0 = output.stridesOf()[0];
            const Nd4jLong zStride1 = output.stridesOf()[dimIC];
            const Nd4jLong zStride2 = output.stridesOf()[dimIH];
            const Nd4jLong zStride3 = output.stridesOf()[dimIH + 1];

            // loop through output array
            auto func = PRAGMA_THREADS_FOR_3D {
                uint xCoord2, xCoord3;
                for (uint b = start_x; b < stop_x; b += inc_x) {
                    for (uint c = start_y; c < stop_y; c += inc_y) {
                        for (uint h = start_z; h < stop_z; h += inc_z) {
                            for (uint w = 0; w < oW; ++w) {
                                xCoord2 = h / factorH;
                                xCoord3 = w / factorW;

                                z[b * zStride0 + c * zStride1 + h * zStride2 + w * zStride3] = x[b * xStride0 + c * xStride1 + xCoord2 * xStride2 + xCoord3 * xStride3];
                            }
                        }
                    }
                }
            };

            samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oH, 1);
        }

//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void upsampling3d_(const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {
            // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
            // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)

            const T* x = input.bufferAsT<T>();
                  T* z = output.bufferAsT<T>();

            const uint dimID = isNCDHW ? 2 : 1;
            const uint dimIC = isNCDHW ? 1 : 4;

            const uint bS = input.sizeAt(0);
            const uint iC = input.sizeAt(dimIC);
            const uint oD = output.sizeAt(dimID);
            const uint oH = output.sizeAt(dimID + 1);
            const uint oW = output.sizeAt(dimID + 2);

            const Nd4jLong xStride0 = input.stridesOf()[0];
            const Nd4jLong xStride1 = input.stridesOf()[dimIC];
            const Nd4jLong xStride2 = input.stridesOf()[dimID];
            const Nd4jLong xStride3 = input.stridesOf()[dimID + 1];
            const Nd4jLong xStride4 = input.stridesOf()[dimID + 2];

            const Nd4jLong zStride0 = output.stridesOf()[0];
            const Nd4jLong zStride1 = output.stridesOf()[dimIC];
            const Nd4jLong zStride2 = output.stridesOf()[dimID];
            const Nd4jLong zStride3 = output.stridesOf()[dimID + 1];
            const Nd4jLong zStride4 = output.stridesOf()[dimID + 2];

            // loop through output array
            auto func = PRAGMA_THREADS_FOR_3D {
                uint xCoord2, xCoord3, xCoord4;

                for (uint b = start_x; b < stop_x; b += inc_x) {
                    for (uint c = start_y; c < stop_y; c += inc_y) {
                        for (uint d = start_z; d < stop_z; d += inc_z) {
                            for (uint h = 0; h < oH; ++h) {
                                for (uint w = 0; w < oW; ++w) {

                                    xCoord2 = d / factorD;
                                    xCoord3 = h / factorH;
                                    xCoord4 = w / factorW;

                                    z[b * zStride0 + c * zStride1 + d * zStride2 + h * zStride3 + w * zStride4] = x[
                                            b * xStride0 + c * xStride1 + xCoord2 * xStride2 + xCoord3 * xStride3 +
                                            xCoord4 * xStride4];
                                }
                            }
                        }
                    }
                }
            };

            samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
        }

//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void upsampling2dBP_(const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
            // gradO has shape [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
            // gradI has shape [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)

            const T* x = gradO.bufferAsT<T>();
                  T* z = gradI.bufferAsT<T>();

            const uint dimIH = isNCHW ? 2 : 1;
            const uint dimIC = isNCHW ? 1 : 3;

            const uint bS = gradI.sizeAt(0);
            const uint iC = gradI.sizeAt(dimIC);
            const uint iH = gradI.sizeAt(dimIH);
            const uint iW = gradI.sizeAt(dimIH + 1);

            const uint factorH = gradO.sizeAt(dimIH)     / iH;
            const uint factorW = gradO.sizeAt(dimIH + 1) / iW;

            const Nd4jLong xStride0 = gradO.stridesOf()[0];
            const Nd4jLong xStride1 = gradO.stridesOf()[dimIC];
            const Nd4jLong xStride2 = gradO.stridesOf()[dimIH];
            const Nd4jLong xStride3 = gradO.stridesOf()[dimIH + 1];

            const Nd4jLong zStride0 = gradI.stridesOf()[0];
            const Nd4jLong zStride1 = gradI.stridesOf()[dimIC];
            const Nd4jLong zStride2 = gradI.stridesOf()[dimIH];
            const Nd4jLong zStride3 = gradI.stridesOf()[dimIH + 1];

            // loop through output array
            auto func = PRAGMA_THREADS_FOR_3D {
                for (uint b = start_x; b < stop_x; b += inc_x) {
                    for (uint c = start_y; c < stop_y; c += inc_y) {
                        for (uint h = start_z; h < stop_z; h += inc_z) {
                            for (uint w = 0; w < iW; ++w) {

                                const auto zOffset = b * zStride0 + c * zStride1 + h * zStride2 + w * zStride3;

                                z[zOffset] = 0;

                                for (uint xh = h * factorH; xh < h * factorH + factorH; ++xh)
                                    for (uint xw = w * factorW; xw < w * factorW + factorW; ++xw)
                                        z[zOffset] += x[b * xStride0 + c * xStride1 + xh * xStride2 + xw * xStride3];
                            }
                        }
                    }
                }
            };

            samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, iH, 1);
        }

//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void upsampling3dBP_(const NDArray& gradO, NDArray& gradI, const bool isNCDHW) {

            // input  has shape [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC)
            // output has shape [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)

            const T* x = gradO.bufferAsT<T>();
                  T* z = gradI.bufferAsT<T>();

            const uint dimID = isNCDHW ? 2 : 1;
            const uint dimIC = isNCDHW ? 1 : 4;

            const uint bS = gradI.sizeAt(0);
            const uint iC = gradI.sizeAt(dimIC);
            const uint iD = gradI.sizeAt(dimID);
            const uint iH = gradI.sizeAt(dimID + 1);
            const uint iW = gradI.sizeAt(dimID + 2);

            const uint factorD = gradO.sizeAt(dimID)     / iD;
            const uint factorH = gradO.sizeAt(dimID + 1) / iH;
            const uint factorW = gradO.sizeAt(dimID + 2) / iW;

            const Nd4jLong xStride0 = gradO.stridesOf()[0];
            const Nd4jLong xStride1 = gradO.stridesOf()[dimIC];
            const Nd4jLong xStride2 = gradO.stridesOf()[dimID];
            const Nd4jLong xStride3 = gradO.stridesOf()[dimID + 1];
            const Nd4jLong xStride4 = gradO.stridesOf()[dimID + 2];

            const Nd4jLong zStride0 = gradI.stridesOf()[0];
            const Nd4jLong zStride1 = gradI.stridesOf()[dimIC];
            const Nd4jLong zStride2 = gradI.stridesOf()[dimID];
            const Nd4jLong zStride3 = gradI.stridesOf()[dimID + 1];
            const Nd4jLong zStride4 = gradI.stridesOf()[dimID + 2];

            // loop through output array
            auto func = PRAGMA_THREADS_FOR_3D {
                for (uint b = start_x; b < stop_x; b += inc_x) {
                    for (uint c = start_y; c < stop_y; c += inc_y) {
                        for (uint d = start_z; d < stop_z; d += inc_z) {
                            for (uint h = 0; h < iH; ++h) {
                                for (uint w = 0; w < iW; ++w) {

                                    const auto zOffset = b * zStride0 + c * zStride1 + d * zStride2 + h * zStride3 + w * zStride4;

                                    z[zOffset] = 0;

                                    for (uint xd = d * factorD; xd < d * factorD + factorD; ++xd)
                                        for (uint xh = h * factorH; xh < h * factorH + factorH; ++xh)
                                            for (uint xw = w * factorW; xw < w * factorW + factorW; ++xw)
                                                z[zOffset] += x[b * xStride0 + c * xStride1 + xd * xStride2 + xh * xStride3 + xw * xStride4];
                                }
                            }
                        }
                    }
                }
            };

            samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, iD, 1);
        }

//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void pooling2d_(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
            // input is  [bS, iC, iH, iW]
            // output is [bS, iC, oH, oW]
            T* out = output.bufferAsT<T>();
            T* in  = const_cast<NDArray&>(input).bufferAsT<T>();

            const int kHEff = kH + (kH-1)*(dH-1);
            const int kWEff = kW + (kW-1)*(dW-1);

            const int bS = input.sizeAt(0);
            const int iC = input.sizeAt(1);
            const int iH = input.sizeAt(2);
            const int iW = input.sizeAt(3);
            const int oC = output.sizeAt(1);
            const int oH = output.sizeAt(2);
            const int oW = output.sizeAt(3);

            nd4j_debug("MKL-DNN is not used for pooling2d!\n", 0);

            const Nd4jLong iStride0 = input.stridesOf()[0];
            const Nd4jLong iStride1 = input.stridesOf()[1];
            const Nd4jLong iStride2 = input.stridesOf()[2];
            const Nd4jLong iStride3 = input.stridesOf()[3];
            const Nd4jLong oStride0 = output.stridesOf()[0];
            const Nd4jLong oStride1 = output.stridesOf()[1];
            const Nd4jLong oStride2 = output.stridesOf()[2];
            const Nd4jLong oStride3 = output.stridesOf()[3];

            const Nd4jLong iStep2   = dH*iStride2;
            const Nd4jLong iStep3   = dW*iStride3;
            const int kProd         = kH*kW;

            if(poolingMode == 0) {        // max
                auto func = PRAGMA_THREADS_FOR_2D {
                    Nd4jLong hstart, wstart, hend, wend;
                    T *pIn;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int oh = 0; oh < oH; ++oh) {
                                for (int ow = 0; ow < oW; ++ow) {

                                    pIn = in + b * iStride0 + c * iStride1;

                                    hstart = oh * sH - pH;
                                    wstart = ow * sW - pW;
                                    hend = hstart + kHEff;
                                    wend = wstart + kWEff;

                                    if (hstart < 0)
                                        hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    hstart *= iStride2;
                                    hend *= iStride2;
                                    wstart *= iStride3;
                                    wend *= iStride3;

                                    T max = -DataTypeUtils::max<T>();

                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep3) {
                                            T val = pIn[kh + kw];
                                            if (val > max)
                                                max = val;
                                        }
                                    out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = max;
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
            }
/*************************************************************************/
            else if(poolingMode == 1) {      // avg
                auto func = PRAGMA_THREADS_FOR_2D {
                    Nd4jLong hstart, wstart, hend, wend;
                    T *pIn;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int oh = 0; oh < oH; ++oh) {
                                for (int ow = 0; ow < oW; ++ow) {

                                    pIn = in + b * iStride0 + c * iStride1;

                                    hstart = oh * sH - pH;
                                    wstart = ow * sW - pW;
                                    hend = hstart + kHEff;
                                    wend = wstart + kWEff;

                                    if (hstart < 0)
                                        hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    hstart *= iStride2;
                                    hend *= iStride2;
                                    wstart *= iStride3;
                                    wend *= iStride3;

                                    T sum = static_cast<T>(0.f);

                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                            sum += pIn[kh + kw];

                                    if (extraParam0 == 0) {                     //Exclude padding
                                        int a = (hend - hstart) / iStep2 + ((hend - hstart) % iStep2 == 0 ? 0 : 1);
                                        int r = (wend - wstart) / iStep3 + ((wend - wstart) % iStep3 == 0 ? 0 : 1);
                                        sum /= static_cast<T>(a * r);          //  Accounts for dilation
                                    } else if (extraParam0 == 1)                  //Include padding
                                        sum /= kProd;

                                    out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
            }
/*************************************************************************/
            else if(poolingMode == 2) {  // pnorm
                auto func = PRAGMA_THREADS_FOR_2D {
                    Nd4jLong hstart, wstart, hend, wend;
                    T *pIn;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int oh = 0; oh < oH; ++oh) {
                                for (int ow = 0; ow < oW; ++ow) {

                                    pIn = in + b * iStride0 + c * iStride1;

                                    hstart = oh * sH - pH;
                                    wstart = ow * sW - pW;
                                    hend = hstart + kHEff;
                                    wend = wstart + kWEff;

                                    if (hstart < 0)
                                        hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    hstart *= iStride2;
                                    hend *= iStride2;
                                    wstart *= iStride3;
                                    wend *= iStride3;

                                    T sum = static_cast<T>(0.f);

                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                            sum += nd4j::math::nd4j_pow<T, T, T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);

                                    sum = nd4j::math::nd4j_pow<T, T, T>(sum, static_cast<T>((T) 1.f) / extraParam0);

                                    out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
            }
            else {
                nd4j_printf("ConvolutionUtils::pooling2d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
                throw "";
            }
        }

//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void pooling3d_(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {
            // input is  [bS, iC, iD, iH, iW]
            // output is [bS, iC, oD, oH, oW]
            T* out = output.bufferAsT<T>();
            T* in  = const_cast<NDArray&>(input).bufferAsT<T>();

            const int kDEff = kD + (kD-1)*(dD-1);
            const int kHEff = kH + (kH-1)*(dH-1);
            const int kWEff = kW + (kW-1)*(dW-1);

            const int bS = input.sizeAt(0);
            const int iC = input.sizeAt(1);
            const int iD = input.sizeAt(2);
            const int iH = input.sizeAt(3);
            const int iW = input.sizeAt(4);
            const int oC = output.sizeAt(1);
            const int oD = output.sizeAt(2);
            const int oH = output.sizeAt(3);
            const int oW = output.sizeAt(4);

            nd4j_debug("MKL-DNN is not used for pooling3d!\n", 0);

            const Nd4jLong iStride0 = input.stridesOf()[0];
            const Nd4jLong iStride1 = input.stridesOf()[1];
            const Nd4jLong iStride2 = input.stridesOf()[2];
            const Nd4jLong iStride3 = input.stridesOf()[3];
            const Nd4jLong iStride4 = input.stridesOf()[4];
            const Nd4jLong oStride0 = output.stridesOf()[0];
            const Nd4jLong oStride1 = output.stridesOf()[1];
            const Nd4jLong oStride2 = output.stridesOf()[2];
            const Nd4jLong oStride3 = output.stridesOf()[3];
            const Nd4jLong oStride4 = output.stridesOf()[4];
            const Nd4jLong iStep2   = dD*iStride2;
            const Nd4jLong iStep3   = dH*iStride3;
            const Nd4jLong iStep4   = dW*iStride4;
            const int kProd         = kD*kH*kW;

            if(poolingMode == 0) {        // max
                auto func = PRAGMA_THREADS_FOR_3D {
                    Nd4jLong dstart, hstart, wstart, dend, hend, wend;
                    T sum, *pIn;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int od = start_z; od < stop_z; od += inc_z) {
                                for (int oh = 0; oh < oH; ++oh) {
                                    for (int ow = 0; ow < oW; ++ow) {

                                        pIn = in + b * iStride0 + c * iStride1;

                                        dstart = od * sD - pD;
                                        hstart = oh * sH - pH;
                                        wstart = ow * sW - pW;
                                        dend = dstart + kDEff;
                                        hend = hstart + kHEff;
                                        wend = wstart + kWEff;

                                        if (dstart < 0)
                                            dstart += dD * ((-dstart + dD - 1) / dD);
                                        if (hstart < 0)
                                            hstart += dH * ((-hstart + dH - 1) / dH);
                                        if (wstart < 0)
                                            wstart += dW * ((-wstart + dW - 1) / dW);
                                        if (dend > iD)
                                            dend -= dD * ((dend - iD + dD - 1) / dD);
                                        if (hend > iH)
                                            hend -= dH * ((hend - iH + dH - 1) / dH);
                                        if (wend > iW)
                                            wend -= dW * ((wend - iW + dW - 1) / dW);

                                        dstart *= iStride2;
                                        dend *= iStride2;
                                        hstart *= iStride3;
                                        hend *= iStride3;
                                        wstart *= iStride4;
                                        wend *= iStride4;

                                        sum = -DataTypeUtils::max<T>();

                                        for (Nd4jLong kd = dstart; kd < dend; kd += iStep2)
                                            for (Nd4jLong kh = hstart; kh < hend; kh += iStep3)
                                                for (Nd4jLong kw = wstart; kw < wend; kw += iStep4) {
                                                    T val = pIn[kd + kh + kw];
                                                    if (val > sum)
                                                        sum = val;
                                                }

                                        out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                                    }
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
            }
/*************************************************************************/
            else if(poolingMode == 1) {     // avg
                auto func = PRAGMA_THREADS_FOR_3D {
                    Nd4jLong dstart, hstart, wstart, dend, hend, wend;
                    T sum, *pIn;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int od = start_z; od < stop_z; od += inc_z) {
                                for (int oh = 0; oh < oH; ++oh) {
                                    for (int ow = 0; ow < oW; ++ow) {

                                        pIn = in + b * iStride0 + c * iStride1;

                                        dstart = od * sD - pD;
                                        hstart = oh * sH - pH;
                                        wstart = ow * sW - pW;
                                        dend = dstart + kDEff;
                                        hend = hstart + kHEff;
                                        wend = wstart + kWEff;

                                        if (dstart < 0)
                                            dstart += dD * ((-dstart + dD - 1) / dD);
                                        if (hstart < 0)
                                            hstart += dH * ((-hstart + dH - 1) / dH);
                                        if (wstart < 0)
                                            wstart += dW * ((-wstart + dW - 1) / dW);
                                        if (dend > iD)
                                            dend -= dD * ((dend - iD + dD - 1) / dD);
                                        if (hend > iH)
                                            hend -= dH * ((hend - iH + dH - 1) / dH);
                                        if (wend > iW)
                                            wend -= dW * ((wend - iW + dW - 1) / dW);

                                        dstart *= iStride2;
                                        dend *= iStride2;
                                        hstart *= iStride3;
                                        hend *= iStride3;
                                        wstart *= iStride4;
                                        wend *= iStride4;

                                        sum = static_cast<T>(0.);

                                        for (Nd4jLong kd = dstart; kd < dend; kd += iStep2)
                                            for (Nd4jLong kh = hstart; kh < hend; kh += iStep3)
                                                for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                                    sum += pIn[kd + kh + kw];

                                        if (extraParam0 == 0)         //Exclude padding
                                            sum /= nd4j::math::nd4j_ceil<double, T>(static_cast<double>(dend - dstart) / static_cast<double>(iStep2)) * nd4j::math::nd4j_ceil<double, T>(static_cast<double>(hend - hstart) / static_cast<double>(iStep3)) * nd4j::math::nd4j_ceil<double, T>(static_cast<double>(wend - wstart) / static_cast<double>(iStep4));   //Accounts for dilation
                                        else if (extraParam0 == 1)    //Include padding
                                            sum /= kProd;

                                        out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                                    }
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
            }
/*************************************************************************/
            else if(poolingMode == 2) {  // pnorm
                auto func = PRAGMA_THREADS_FOR_3D {
                    Nd4jLong dstart, hstart, wstart, dend, hend, wend;
                    T sum, *pIn;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int od = start_z; od < stop_z; od += inc_z) {
                                for (int oh = 0; oh < oH; ++oh) {
                                    for (int ow = 0; ow < oW; ++ow) {

                                        pIn = in + b * iStride0 + c * iStride1;

                                        dstart = od * sD - pD;
                                        hstart = oh * sH - pH;
                                        wstart = ow * sW - pW;
                                        dend = dstart + kDEff;
                                        hend = hstart + kHEff;
                                        wend = wstart + kWEff;

                                        if (dstart < 0)
                                            dstart += dD * ((-dstart + dD - 1) / dD);
                                        if (hstart < 0)
                                            hstart += dH * ((-hstart + dH - 1) / dH);
                                        if (wstart < 0)
                                            wstart += dW * ((-wstart + dW - 1) / dW);
                                        if (dend > iD)
                                            dend -= dD * ((dend - iD + dD - 1) / dD);
                                        if (hend > iH)
                                            hend -= dH * ((hend - iH + dH - 1) / dH);
                                        if (wend > iW)
                                            wend -= dW * ((wend - iW + dW - 1) / dW);

                                        dstart *= iStride2;
                                        dend *= iStride2;
                                        hstart *= iStride3;
                                        hend *= iStride3;
                                        wstart *= iStride4;
                                        wend *= iStride4;

                                        sum = static_cast<T>(0.);

                                        for (Nd4jLong kd = dstart; kd < dend; kd += iStep2)
                                            for (Nd4jLong kh = hstart; kh < hend; kh += iStep3)
                                                for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                                    sum += nd4j::math::nd4j_pow<T, T, T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0);

                                        sum = nd4j::math::nd4j_pow<T, T, T>(sum, (T) 1.f / extraParam0);

                                        out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
                                    }
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
            }
            else {
                nd4j_printf("ConvolutionUtils::pooling3d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
                throw std::runtime_error("Incorrect poooling3d mode");
            }
        }


//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void pooling2dBP_(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
            // input [bS, iC, iH, iW]
            // gradI [bS, iC, iH, iW] -> gradI is output in this function
            // gradO [bS, iC, oH, oW]

            // initial zeroing of gradI
            gradI.nullify();

            T* in = const_cast<NDArray&>(input).bufferAsT<T>();
            T* gO = const_cast<NDArray&>(gradO).bufferAsT<T>();
            T* gI = gradI.bufferAsT<T>();

            const int kHEff = kH + (kH-1)*(dH-1);
            const int kWEff = kW + (kW-1)*(dW-1);

            const int bS = gradI.sizeAt(0);
            const int iC = gradI.sizeAt(1);
            const int iH = gradI.sizeAt(2);
            const int iW = gradI.sizeAt(3);
            const int oC = gradO.sizeAt(1);
            const int oH = gradO.sizeAt(2);
            const int oW = gradO.sizeAt(3);

            nd4j_debug("MKL-DNN is not used for pooling2d_bp!\n", 0);

            const Nd4jLong iStride0  = input.stridesOf()[0];
            const Nd4jLong iStride1  = input.stridesOf()[1];
            const Nd4jLong iStride2  = input.stridesOf()[2];
            const Nd4jLong iStride3  = input.stridesOf()[3];
            const Nd4jLong gIStride0 = gradI.stridesOf()[0];
            const Nd4jLong gIStride1 = gradI.stridesOf()[1];
            const Nd4jLong gIStride2 = gradI.stridesOf()[2];
            const Nd4jLong gIStride3 = gradI.stridesOf()[3];
            const Nd4jLong oStride0  = gradO.stridesOf()[0];
            const Nd4jLong oStride1  = gradO.stridesOf()[1];
            const Nd4jLong oStride2  = gradO.stridesOf()[2];
            const Nd4jLong oStride3  = gradO.stridesOf()[3];
            const Nd4jLong iStep2    = dH*iStride2;
            const Nd4jLong iStep3    = dW*iStride3;
            const Nd4jLong gIStep2   = dH*gIStride2;
            const Nd4jLong gIStep3   = dW*gIStride3;
            const int      kProd     = kH*kW;

            const bool sameStrides = iStride0 == gIStride0 && iStride1 == gIStride1 && iStride2 == gIStride2 && iStride3 == gIStride3;

            if(poolingMode == 0) {        // max
                auto func = PRAGMA_THREADS_FOR_2D {
                    Nd4jLong hstart, wstart,hend, wend, maxKH, maxKW;
                    T sum, valO, *pIn, *pgI;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int oh = 0; oh < oH; ++oh) {
                                for (int ow = 0; ow < oW; ++ow) {

                                    pIn = in + b * iStride0 + c * iStride1;

                                    hstart = oh * sH - pH;
                                    wstart = ow * sW - pW;
                                    hend = hstart + kHEff;
                                    wend = wstart + kWEff;

                                    if (hstart < 0)
                                        hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) / dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) / dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    sum = -DataTypeUtils::max<T>();
                                    valO = gO[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3];

                                    if (sameStrides) {

                                        hstart *= iStride2;
                                        hend *= iStride2;
                                        wstart *= iStride3;
                                        wend *= iStride3;

                                        // we set these to default values
                                        maxKH = hstart;
                                        maxKW = wstart;

                                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3) {
                                                T valIn = pIn[kh + kw];
                                                if (valIn > sum) {
                                                    sum = valIn;
                                                    maxKH = kh;
                                                    maxKW = kw;
                                                }
                                            }
                                        gI[pIn - in + maxKH + maxKW] += valO;
                                    } else {

                                        // we set these to default values
                                        maxKH = hstart;
                                        maxKW = wstart;

                                        for (Nd4jLong kh = hstart; kh < hend; kh += dH)
                                            for (Nd4jLong kw = wstart; kw < wend; kw += dW) {
                                                T valIn = pIn[kh * iStride2 + kw * iStride3];
                                                if (valIn > sum) {
                                                    sum = valIn;
                                                    maxKH = kh;
                                                    maxKW = kw;
                                                }
                                            }

                                        gI[b * gIStride0 + c * gIStride1 + maxKH * gIStride2 + maxKW * gIStride3] += valO;
                                    }
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
            }
/*************************************************************************/
            else if(poolingMode == 1) {     // avg
                auto func = PRAGMA_THREADS_FOR_2D {
                    Nd4jLong hstart, wstart, hend, wend, maxKH, maxKW;
                    T sum, valO, *pIn, *pgI;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int oh = 0; oh < oH; ++oh) {
                                for (int ow = 0; ow < oW; ++ow) {

                                    pgI = gI + b * gIStride0 + c * gIStride1;

                                    hstart = oh * sH - pH;
                                    wstart = ow * sW - pW;
                                    hend = hstart + kHEff;
                                    wend = wstart + kWEff;

                                    if (hstart < 0)
                                        hstart += dH * ((-hstart + dH - 1) /
                                                        dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) /
                                                        dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) /
                                                      dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) /
                                                      dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    hstart *= gIStride2;
                                    hend *= gIStride2;
                                    wstart *= gIStride3;
                                    wend *= gIStride3;

                                    valO = gO[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3];

                                    if ((int) extraParam0 == 0)         //Exclude padding
                                        valO /= static_cast<T>(nd4j::math::nd4j_ceil<double, T>(
                                                static_cast<double>(hend - hstart) / static_cast<double>(gIStep2))) *
                                                static_cast<T>(nd4j::math::nd4j_ceil<double, T>(
                                                        static_cast<double>(wend - wstart) /
                                                        static_cast<double>(gIStep3)));   //Accounts for dilation
                                    else if ((int) extraParam0 == 1)    //Include padding
                                        valO /= kProd;

                                    for (Nd4jLong kh = hstart; kh < hend; kh += gIStep2)
                                        for (Nd4jLong kw = wstart; kw < wend; kw += gIStep3)
                                            pgI[kh + kw] += valO;
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
            }
/*************************************************************************/
            else if(poolingMode == 2) {  // pnorm
                auto func = PRAGMA_THREADS_FOR_2D {
                    Nd4jLong hstart, wstart, hend, wend, maxKH, maxKW;
                    T sum, valO, *pIn, *pgI;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int oh = 0; oh < oH; ++oh) {
                                for (int ow = 0; ow < oW; ++ow) {

                                    pIn = in + b * iStride0 + c * iStride1;
                                    pgI = sameStrides ? gI + (pIn - in) : gI + b * gIStride0 + c * gIStride1;

                                    hstart = oh * sH - pH;
                                    wstart = ow * sW - pW;
                                    hend = hstart + kHEff;
                                    wend = wstart + kWEff;

                                    if (hstart < 0)
                                        hstart += dH * ((-hstart + dH - 1) /
                                                        dH); // (Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) /
                                                        dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) /
                                                      dH); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) /
                                                      dW); //(Nd4jLong)nd4j::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    sum = static_cast<T>(0.f);
                                    valO = gO[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3];

                                    if (sameStrides) {

                                        hstart *= iStride2;
                                        hend *= iStride2;
                                        wstart *= iStride3;
                                        wend *= iStride3;

                                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                                sum += nd4j::math::nd4j_pow<T, T, T>(
                                                        nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);

                                        valO *= nd4j::math::nd4j_pow<T, T, T>(sum,
                                                                              ((T) 1. - extraParam0) / extraParam0);

                                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                                pgI[kh + kw] += valO * nd4j::math::nd4j_pow<T, T, T>(
                                                        nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0 - 1.f) *
                                                                nd4j::math::nd4j_sgn<T, T>(pIn[kh + kw]);
                                    } else {

                                        for (Nd4jLong kh = hstart; kh < hend; kh += dH)
                                            for (Nd4jLong kw = wstart; kw < wend; kw += dW)
                                                sum += nd4j::math::nd4j_pow<T, T, T>(
                                                        nd4j::math::nd4j_abs<T>(pIn[kh * iStride2 + kw * iStride3]),
                                                        extraParam0);

                                        valO *= nd4j::math::nd4j_pow<T, T, T>(sum,
                                                                              ((T) 1. - extraParam0) / extraParam0);

                                        for (Nd4jLong kh = hstart; kh < hend; kh += dH) {
                                            for (Nd4jLong kw = wstart; kw < wend; kw += dW) {
                                                const auto inVal = pIn[kh * iStride2 + kw * iStride3];
                                                pgI[kh * gIStride2 + kw * gIStride3] += valO *
                                                                                        nd4j::math::nd4j_pow<T, T, T>(
                                                                                                nd4j::math::nd4j_abs<T>(
                                                                                                        inVal),
                                                                                                extraParam0 - 1.f) *
                                                                                        nd4j::math::nd4j_sgn<T, T>(
                                                                                                inVal);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
            }
            else {
                nd4j_printf("ConvolutionUtils::pooling2dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
                throw std::runtime_error("Incorrect pooling2dBP mode");
            }
        }

//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void pooling3dBP_(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {
            // input [bS, iC, iD, iH, iW]
            // gradI [bS, iC, iD, iH, iW] -> gradI is output in this function
            // gradO [bS, iC, oD, oH, oW]

            // initial zeroing of gradI
            gradI.nullify();

            T* in = const_cast<NDArray&>(input).bufferAsT<T>();
            T* gO = const_cast<NDArray&>(gradO).bufferAsT<T>();
            T* gI = gradI.bufferAsT<T>();

            const int kDEff = kD + (kD-1)*(dD-1);
            const int kHEff = kH + (kH-1)*(dH-1);
            const int kWEff = kW + (kW-1)*(dW-1);

            const int bS = gradI.sizeAt(0);
            const int iC = gradI.sizeAt(1);
            const int iD = gradI.sizeAt(2);
            const int iH = gradI.sizeAt(3);
            const int iW = gradI.sizeAt(4);
            const int oC = gradO.sizeAt(1);
            const int oD = gradO.sizeAt(2);
            const int oH = gradO.sizeAt(3);
            const int oW = gradO.sizeAt(4);

            nd4j_debug("MKL-DNN is not used for pooling3d_bp!\n", 0);

            const Nd4jLong iStride0  = input.stridesOf()[0];
            const Nd4jLong iStride1  = input.stridesOf()[1];
            const Nd4jLong iStride2  = input.stridesOf()[2];
            const Nd4jLong iStride3  = input.stridesOf()[3];
            const Nd4jLong iStride4  = input.stridesOf()[4];
            const Nd4jLong gIStride0 = gradI.stridesOf()[0];
            const Nd4jLong gIStride1 = gradI.stridesOf()[1];
            const Nd4jLong gIStride2 = gradI.stridesOf()[2];
            const Nd4jLong gIStride3 = gradI.stridesOf()[3];
            const Nd4jLong gIStride4 = gradI.stridesOf()[4];
            const Nd4jLong oStride0 = gradO.stridesOf()[0];
            const Nd4jLong oStride1 = gradO.stridesOf()[1];
            const Nd4jLong oStride2 = gradO.stridesOf()[2];
            const Nd4jLong oStride3 = gradO.stridesOf()[3];
            const Nd4jLong oStride4 = gradO.stridesOf()[4];
            const Nd4jLong iStep2   = dD*iStride2;
            const Nd4jLong iStep3   = dH*iStride3;
            const Nd4jLong iStep4   = dW*iStride4;
            const Nd4jLong gIStep2  = dD*gIStride2;
            const Nd4jLong gIStep3  = dH*gIStride3;
            const Nd4jLong gIStep4  = dW*gIStride4;
            const int      kProd    = kD*kH*kW;

            const bool sameStrides = iStride0 == gIStride0 && iStride1 == gIStride1 && iStride2 == gIStride2 && iStride3 == gIStride3 && iStride4 == gIStride4;

            if(poolingMode == 0) {        // max
                auto func = PRAGMA_THREADS_FOR_3D {
                    Nd4jLong dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW;
                    T sum, valO, *pIn, *pgI;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int od = start_z; od < stop_z; od += inc_z) {
                                for (int oh = 0; oh < oH; ++oh) {
                                    for (int ow = 0; ow < oW; ++ow) {

                                        pIn = in + b * iStride0 + c * iStride1;

                                        dstart = od * sD - pD;
                                        hstart = oh * sH - pH;
                                        wstart = ow * sW - pW;
                                        dend = dstart + kDEff;
                                        hend = hstart + kHEff;
                                        wend = wstart + kWEff;

                                        if (dstart < 0)
                                            dstart += dD * ((-dstart + dD - 1) / dD);
                                        if (hstart < 0)
                                            hstart += dH * ((-hstart + dH - 1) / dH);
                                        if (wstart < 0)
                                            wstart += dW * ((-wstart + dW - 1) / dW);
                                        if (dend > iD)
                                            dend -= dD * ((dend - iD + dD - 1) / dD);
                                        if (hend > iH)
                                            hend -= dH * ((hend - iH + dH - 1) / dH);
                                        if (wend > iW)
                                            wend -= dW * ((wend - iW + dW - 1) / dW);

                                        sum = -DataTypeUtils::max<T>();
                                        valO = gO[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4];

                                        if (sameStrides) {

                                            dstart *= iStride2;
                                            dend *= iStride2;
                                            hstart *= iStride3;
                                            hend *= iStride3;
                                            wstart *= iStride4;
                                            wend *= iStride4;

                                            maxKD = dstart;
                                            maxKH = hstart;
                                            maxKW = wstart;

                                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2)
                                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3)
                                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4) {
                                                        T valIn = pIn[kd + kh + kw];
                                                        if (valIn > sum) {
                                                            sum = valIn;
                                                            maxKD = kd;
                                                            maxKH = kh;
                                                            maxKW = kw;
                                                        }
                                                    }
                                            gI[pIn - in + maxKD + maxKH + maxKW] += valO;
                                        } else {

                                            // we set these to default values
                                            maxKH = hstart;
                                            maxKW = wstart;
                                            maxKD = dstart;

                                            for (Nd4jLong kd = dstart; kd < dend; kd += dD)
                                                for (Nd4jLong kh = hstart; kh < hend; kh += dH)
                                                    for (Nd4jLong kw = wstart; kw < wend; kw += dW) {
                                                        T valIn = pIn[kd * iStride2 + kh * iStride3 + kw * iStride4];
                                                        if (valIn > sum) {
                                                            sum = valIn;
                                                            maxKD = kd;
                                                            maxKH = kh;
                                                            maxKW = kw;
                                                        }
                                                    }

                                            gI[b * gIStride0 + c * gIStride1 + maxKD * gIStride2 + maxKH * gIStride3 + maxKW * gIStride4] += valO;
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
            }
/*************************************************************************/
            else if(poolingMode == 1) {     // avg
                auto func = PRAGMA_THREADS_FOR_3D {
                    Nd4jLong dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW;
                    T sum, valO, *pIn, *pgI;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int od = start_z; od < stop_z; od += inc_z) {
                                for (int oh = 0; oh < oH; ++oh) {
                                    for (int ow = 0; ow < oW; ++ow) {

                                        pgI = gI + b * gIStride0 + c * gIStride1;

                                        dstart = od * sD - pD;
                                        hstart = oh * sH - pH;
                                        wstart = ow * sW - pW;
                                        dend = dstart + kDEff;
                                        hend = hstart + kHEff;
                                        wend = wstart + kWEff;

                                        if (dstart < 0)
                                            dstart += dD * ((-dstart + dD - 1) / dD);
                                        if (hstart < 0)
                                            hstart += dH * ((-hstart + dH - 1) / dH);
                                        if (wstart < 0)
                                            wstart += dW * ((-wstart + dW - 1) / dW);
                                        if (dend > iD)
                                            dend -= dD * ((dend - iD + dD - 1) / dD);
                                        if (hend > iH)
                                            hend -= dH * ((hend - iH + dH - 1) / dH);
                                        if (wend > iW)
                                            wend -= dW * ((wend - iW + dW - 1) / dW);

                                        dstart *= gIStride2;
                                        dend *= gIStride2;
                                        hstart *= gIStride3;
                                        hend *= gIStride3;
                                        wstart *= gIStride4;
                                        wend *= gIStride4;

                                        valO = gO[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4];

                                        if (extraParam0 == 0)         //Exclude padding
                                            valO /= nd4j::math::nd4j_ceil<double, T>(static_cast<double>(dend - dstart) / static_cast<double>(gIStep2)) * nd4j::math::nd4j_ceil<double, T>(static_cast<double>(hend - hstart) / static_cast<double>(gIStep3)) * nd4j::math::nd4j_ceil<double, T>(static_cast<double>(wend - wstart) / static_cast<double>(gIStep4));   //Accounts for dilation
                                        else if (extraParam0 == 1)    //Include padding
                                            valO /= kProd;

                                        for (Nd4jLong kd = dstart; kd < dend; kd += gIStep2)
                                            for (Nd4jLong kh = hstart; kh < hend; kh += gIStep3)
                                                for (Nd4jLong kw = wstart; kw < wend; kw += gIStep4)
                                                    pgI[kd + kh + kw] += valO;
                                    }
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
            }
/*************************************************************************/
            else if(poolingMode == 2) {  // pnorm
                auto func = PRAGMA_THREADS_FOR_3D {
                    Nd4jLong dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW;
                    T sum, valO, *pIn, *pgI;

                    for (int b = start_x; b < stop_x; b += inc_x) {
                        for (int c = start_y; c < stop_y; c += inc_y) {
                            for (int od = start_z; od < stop_z; od += inc_z) {
                                for (int oh = 0; oh < oH; ++oh) {
                                    for (int ow = 0; ow < oW; ++ow) {

                                        pIn = in + b * iStride0 + c * iStride1;
                                        pgI = gI + (pIn - in);

                                        dstart = od * sD - pD;
                                        hstart = oh * sH - pH;
                                        wstart = ow * sW - pW;
                                        dend = dstart + kDEff;
                                        hend = hstart + kHEff;
                                        wend = wstart + kWEff;

                                        if (dstart < 0)
                                            dstart += dD * ((-dstart + dD - 1) / dD);
                                        if (hstart < 0)
                                            hstart += dH * ((-hstart + dH - 1) / dH);
                                        if (wstart < 0)
                                            wstart += dW * ((-wstart + dW - 1) / dW);
                                        if (dend > iD)
                                            dend -= dD * ((dend - iD + dD - 1) / dD);
                                        if (hend > iH)
                                            hend -= dH * ((hend - iH + dH - 1) / dH);
                                        if (wend > iW)
                                            wend -= dW * ((wend - iW + dW - 1) / dW);

                                        sum = static_cast<T>(0.);
                                        valO = gO[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4];

                                        if (sameStrides) {

                                            dstart *= iStride2;
                                            dend *= iStride2;
                                            hstart *= iStride3;
                                            hend *= iStride3;
                                            wstart *= iStride4;
                                            wend *= iStride4;

                                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2)
                                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3)
                                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                                        sum += nd4j::math::nd4j_pow<T, T, T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]), extraParam0);

                                            valO *= nd4j::math::nd4j_pow<T, T, T>(sum, ((T) 1.f - extraParam0) / extraParam0);

                                            for (Nd4jLong kd = dstart; kd < dend; kd += iStep2)
                                                for (Nd4jLong kh = hstart; kh < hend; kh += iStep3)
                                                    for (Nd4jLong kw = wstart; kw < wend; kw += iStep4)
                                                        pgI[kd + kh + kw] += valO * nd4j::math::nd4j_pow<T, T, T>(nd4j::math::nd4j_abs<T>(pIn[kd + kh + kw]),extraParam0 - (T) 1.f) * nd4j::math::nd4j_sgn<T, T>(pIn[kd + kh + kw]);
                                        } else {
                                            for (Nd4jLong kd = dstart; kd < dend; kd += dD)
                                                for (Nd4jLong kh = hstart; kh < hend; kh += dH)
                                                    for (Nd4jLong kw = wstart; kw < wend; kw += dW)
                                                        sum += nd4j::math::nd4j_pow<T, T, T>(nd4j::math::nd4j_abs<T>(pIn[kd * iStride2 + kh * iStride3 + kw * iStride4]), extraParam0);

                                            valO *= nd4j::math::nd4j_pow<T, T, T>(sum, ((T) 1.f - extraParam0) / extraParam0);

                                            for (Nd4jLong kd = dstart; kd < dend; kd += dD)
                                                for (Nd4jLong kh = hstart; kh < hend; kh += dH)
                                                    for (Nd4jLong kw = wstart; kw < wend; kw += dW) {
                                                        const auto inVal = pIn[kD * iStride2 + kh * iStride3 + kw * iStride4];
                                                        pgI[kd * gIStride2 + kh * gIStride3 + kw * gIStride4] += valO * nd4j::math::nd4j_pow<T, T, T>(nd4j::math::nd4j_abs<T>(inVal), extraParam0 - 1.f) * nd4j::math::nd4j_sgn<T, T>(inVal);
                                                    }
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
            }
            else {
                nd4j_printf("ConvolutionUtils::pooling3dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
                throw "";
            }
        }




        void ConvolutionUtils::conv2d(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {
            BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), conv2d_, (block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW), FLOAT_TYPES);
        }
        void ConvolutionUtils::conv2dBP(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {
            BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), conv2dBP_, (block, input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW), FLOAT_TYPES);
        }
        void ConvolutionUtils::depthwiseConv2d(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {
            BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), depthwiseConv2d_, (block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW), FLOAT_TYPES);
        }
        void ConvolutionUtils::depthwiseConv2dBP(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {
            BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), depthwiseConv2dBP_, (input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW), FLOAT_TYPES);
        }
        void ConvolutionUtils::sconv2d(nd4j::graph::Context& block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW) {
            BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), sconv2d_, (block, input, weightsDepth, weightsPoint, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW), FLOAT_TYPES);
        }
        void ConvolutionUtils::vol2col(nd4j::graph::Context& block, const NDArray& volume, NDArray& columns, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
            BUILD_SINGLE_SELECTOR(volume.dataType(), vol2col_, (volume, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);
        }
        void ConvolutionUtils::col2vol(nd4j::graph::Context& block, const NDArray& columns, NDArray& volume, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
            BUILD_SINGLE_SELECTOR(volume.dataType(), col2vol_, (columns, volume, sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);
        }
        void ConvolutionUtils::upsampling2d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW) {
            BUILD_SINGLE_SELECTOR(input.dataType(), upsampling2d_, (input, output, factorH, factorW, isNCHW), FLOAT_TYPES);
        }
        void ConvolutionUtils::upsampling3d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {
            BUILD_SINGLE_SELECTOR(input.dataType(), upsampling3d_, (input, output, factorD, factorH, factorW, isNCDHW), FLOAT_TYPES);
        }
        void ConvolutionUtils::upsampling2dBP(nd4j::graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
            BUILD_SINGLE_SELECTOR(gradO.dataType(), upsampling2dBP_, (gradO, gradI, isNCHW), FLOAT_TYPES);
        }
        void ConvolutionUtils::upsampling3dBP(nd4j::graph::Context& block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {
            BUILD_SINGLE_SELECTOR(gradO.dataType(), upsampling3dBP_, (gradO, gradI, isNCHW), FLOAT_TYPES);
        }



        void ConvolutionUtils::pooling2d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {
            BUILD_SINGLE_SELECTOR(input.dataType(), pooling2d_, (block, input, output, kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
        }
        void ConvolutionUtils::pooling3d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {
            BUILD_SINGLE_SELECTOR(input.dataType(), pooling3d_, (block, input, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
        }
        void ConvolutionUtils::pooling2dBP(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
            BUILD_SINGLE_SELECTOR(input.dataType(), pooling2dBP_, (block, input, gradO, gradI, kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
        }
        void ConvolutionUtils::pooling3dBP(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {
            BUILD_SINGLE_SELECTOR(input.dataType(), pooling3dBP_, (block, input, gradO, gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
        }
    }
}