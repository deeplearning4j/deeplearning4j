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
        template <typename T>
        static void pooling2dBP_(sd::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
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
                                        hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) / dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) / dH); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) / dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

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
                                                        dH); // (Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) /
                                                        dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) /
                                                      dH); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) /
                                                      dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    hstart *= gIStride2;
                                    hend *= gIStride2;
                                    wstart *= gIStride3;
                                    wend *= gIStride3;

                                    valO = gO[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3];

                                    if ((int) extraParam0 == 0)         //Exclude padding
                                        valO /= static_cast<T>(sd::math::nd4j_ceil<double, T>(
                                                static_cast<double>(hend - hstart) / static_cast<double>(gIStep2))) *
                                                static_cast<T>(sd::math::nd4j_ceil<double, T>(
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
                                                        dH); // (Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) /
                                                        dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) /
                                                      dH); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) /
                                                      dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    sum = static_cast<T>(0.f);
                                    valO = gO[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3];

                                    if (sameStrides) {

                                        hstart *= iStride2;
                                        hend *= iStride2;
                                        wstart *= iStride3;
                                        wend *= iStride3;

                                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                                sum += sd::math::nd4j_pow<T, T, T>(
                                                        sd::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);

                                        valO *= sd::math::nd4j_pow<T, T, T>(sum,
                                                                              ((T) 1. - extraParam0) / extraParam0);

                                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                                pgI[kh + kw] += valO * sd::math::nd4j_pow<T, T, T>(
                                                        sd::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0 - 1.f) *
                                                                sd::math::nd4j_sgn<T, T>(pIn[kh + kw]);
                                    } else {

                                        for (Nd4jLong kh = hstart; kh < hend; kh += dH)
                                            for (Nd4jLong kw = wstart; kw < wend; kw += dW)
                                                sum += sd::math::nd4j_pow<T, T, T>(
                                                        sd::math::nd4j_abs<T>(pIn[kh * iStride2 + kw * iStride3]),
                                                        extraParam0);

                                        valO *= sd::math::nd4j_pow<T, T, T>(sum,
                                                                              ((T) 1. - extraParam0) / extraParam0);

                                        for (Nd4jLong kh = hstart; kh < hend; kh += dH) {
                                            for (Nd4jLong kw = wstart; kw < wend; kw += dW) {
                                                const auto inVal = pIn[kh * iStride2 + kw * iStride3];
                                                pgI[kh * gIStride2 + kw * gIStride3] += valO *
                                                                                        sd::math::nd4j_pow<T, T, T>(
                                                                                                sd::math::nd4j_abs<T>(
                                                                                                        inVal),
                                                                                                extraParam0 - 1.f) *
                                                                                        sd::math::nd4j_sgn<T, T>(
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

void ConvolutionUtils::pooling2dBP(sd::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
            BUILD_SINGLE_SELECTOR(input.dataType(), pooling2dBP_, (block, input, gradO, gradI, kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0), FLOAT_TYPES);
        }

}
}
