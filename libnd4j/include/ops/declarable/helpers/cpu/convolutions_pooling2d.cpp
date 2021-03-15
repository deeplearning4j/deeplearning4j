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

#include <ops/declarable/helpers/convolutions.h>
#include <execution/Threads.h>

namespace sd {
    namespace ops  {

//////////////////////////////////////////////////////////////////////////
        template <typename T>
        static void pooling2d_(sd::graph::Context& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {
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
                                        hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) / dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) / dH); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) / dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

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
                                        hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) / dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) / dH); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) / dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

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
                                        hstart += dH * ((-hstart + dH - 1) / dH); // (Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                                    if (wstart < 0)
                                        wstart += dW * ((-wstart + dW - 1) / dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                                    if (hend > iH)
                                        hend -= dH * ((hend - iH + dH - 1) / dH); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                                    if (wend > iW)
                                        wend -= dW * ((wend - iW + dW - 1) / dW); //(Nd4jLong)sd::math::nd4j_ceil<T,T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                                    hstart *= iStride2;
                                    hend *= iStride2;
                                    wstart *= iStride3;
                                    wend *= iStride3;

                                    T sum = static_cast<T>(0.f);

                                    for (Nd4jLong kh = hstart; kh < hend; kh += iStep2)
                                        for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                            sum += sd::math::nd4j_pow<T, T, T>(sd::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);

                                    sum = sd::math::nd4j_pow<T, T, T>(sum, static_cast<T>((T) 1.f) / extraParam0);

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

        void ConvolutionUtils::pooling2d(sd::graph::Context& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {
            BUILD_SINGLE_SELECTOR(input.dataType(), pooling2d_, (block, input, output, kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0), NUMERIC_TYPES);
        }

}
}
