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
static void pooling3dBP_(sd::graph::Context& block, NDArray& input, NDArray& gradO, NDArray& gradI,
                         const LongType kD, const LongType kH, const LongType kW, const LongType sD, const LongType sH, const LongType sW,
                         const LongType pD, const LongType pH, const LongType pW, const LongType dD, const LongType dH, const LongType dW,
                         const int poolingMode, const int extraParam0) {
  // input [bS, iC, iD, iH, iW]
  // gradI [bS, iC, iD, iH, iW] -> gradI is output in this function
  // gradO [bS, iC, oD, oH, oW]

  // initial zeroing of gradI
  gradI.nullify();

  T* in = const_cast<NDArray&>(input).bufferAsT<T>();
  T* gO = const_cast<NDArray&>(gradO).bufferAsT<T>();
  T* gI = gradI.bufferAsT<T>();

  const int kDEff = kD + (kD - 1) * (dD - 1);
  const int kHEff = kH + (kH - 1) * (dH - 1);
  const int kWEff = kW + (kW - 1) * (dW - 1);

  const int bS = gradI.sizeAt(0);
  const int iC = gradI.sizeAt(1);
  const int iD = gradI.sizeAt(2);
  const int iH = gradI.sizeAt(3);
  const int iW = gradI.sizeAt(4);
  const int oC = gradO.sizeAt(1);
  const int oD = gradO.sizeAt(2);
  const int oH = gradO.sizeAt(3);
  const int oW = gradO.sizeAt(4);

  sd_debug("MKL-DNN is not used for pooling3d_bp!\n", 0);

  const sd::LongType iStride0 = input.stridesOf()[0];
  const sd::LongType iStride1 = input.stridesOf()[1];
  const sd::LongType iStride2 = input.stridesOf()[2];
  const sd::LongType iStride3 = input.stridesOf()[3];
  const sd::LongType iStride4 = input.stridesOf()[4];
  const sd::LongType gIStride0 = gradI.stridesOf()[0];
  const sd::LongType gIStride1 = gradI.stridesOf()[1];
  const sd::LongType gIStride2 = gradI.stridesOf()[2];
  const sd::LongType gIStride3 = gradI.stridesOf()[3];
  const sd::LongType gIStride4 = gradI.stridesOf()[4];
  const sd::LongType oStride0 = gradO.stridesOf()[0];
  const sd::LongType oStride1 = gradO.stridesOf()[1];
  const sd::LongType oStride2 = gradO.stridesOf()[2];
  const sd::LongType oStride3 = gradO.stridesOf()[3];
  const sd::LongType oStride4 = gradO.stridesOf()[4];
  const sd::LongType iStep2 = dD * iStride2;
  const sd::LongType iStep3 = dH * iStride3;
  const sd::LongType iStep4 = dW * iStride4;
  const sd::LongType gIStep2 = dD * gIStride2;
  const sd::LongType gIStep3 = dH * gIStride3;
  const sd::LongType gIStep4 = dW * gIStride4;
  const int kProd = kD * kH * kW;

  const bool sameStrides = iStride0 == gIStride0 && iStride1 == gIStride1 && iStride2 == gIStride2 &&
                           iStride3 == gIStride3 && iStride4 == gIStride4;

  if (poolingMode == 0) {  // max
    auto func = PRAGMA_THREADS_FOR_2D {
      sd::LongType dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW;
      T sum, valO, *pIn, *pgI;

      for (int b = start_x; b < stop_x; b++) {
        for (int c = start_y; c < stop_y; c++) {
          for (int od = 0; od < oD; od++) {
            for (int oh = 0; oh < oH; ++oh) {
              for (int ow = 0; ow < oW; ++ow) {
                pIn = in + b * iStride0 + c * iStride1;

                dstart = od * sD - pD;
                hstart = oh * sH - pH;
                wstart = ow * sW - pW;
                dend = dstart + kDEff;
                hend = hstart + kHEff;
                wend = wstart + kWEff;

                if (dstart < 0) dstart += dD * ((-dstart + dD - 1) / dD);
                if (hstart < 0) hstart += dH * ((-hstart + dH - 1) / dH);
                if (wstart < 0) wstart += dW * ((-wstart + dW - 1) / dW);
                if (dend > iD) dend -= dD * ((dend - iD + dD - 1) / dD);
                if (hend > iH) hend -= dH * ((hend - iH + dH - 1) / dH);
                if (wend > iW) wend -= dW * ((wend - iW + dW - 1) / dW);

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

                  for (sd::LongType kd = dstart; kd < dend; kd += iStep2)
                    for (sd::LongType kh = hstart; kh < hend; kh += iStep3)
                      for (sd::LongType kw = wstart; kw < wend; kw += iStep4) {
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

                  for (sd::LongType kd = dstart; kd < dend; kd += dD)
                    for (sd::LongType kh = hstart; kh < hend; kh += dH)
                      for (sd::LongType kw = wstart; kw < wend; kw += dW) {
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

    samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
  }
  /*************************************************************************/
  else if (poolingMode == 1) {  // avg
    auto func = PRAGMA_THREADS_FOR_2D {
      sd::LongType dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW;
      T sum, valO, *pIn, *pgI;

      for (int b = start_x; b < stop_x; b++) {
        for (int c = start_y; c < stop_y; c++) {
          for (int od = 0; od < oD; od++) {
            for (int oh = 0; oh < oH; ++oh) {
              for (int ow = 0; ow < oW; ++ow) {
                pgI = gI + b * gIStride0 + c * gIStride1;

                dstart = od * sD - pD;
                hstart = oh * sH - pH;
                wstart = ow * sW - pW;
                dend = dstart + kDEff;
                hend = hstart + kHEff;
                wend = wstart + kWEff;

                if (dstart < 0) dstart += dD * ((-dstart + dD - 1) / dD);
                if (hstart < 0) hstart += dH * ((-hstart + dH - 1) / dH);
                if (wstart < 0) wstart += dW * ((-wstart + dW - 1) / dW);
                if (dend > iD) dend -= dD * ((dend - iD + dD - 1) / dD);
                if (hend > iH) hend -= dH * ((hend - iH + dH - 1) / dH);
                if (wend > iW) wend -= dW * ((wend - iW + dW - 1) / dW);

                dstart *= gIStride2;
                dend *= gIStride2;
                hstart *= gIStride3;
                hend *= gIStride3;
                wstart *= gIStride4;
                wend *= gIStride4;

                valO = gO[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4];

                if (extraParam0 == 0)  // Exclude padding
                  valO /=
                      sd::math::sd_ceil<double, T>(static_cast<double>(dend - dstart) / static_cast<double>(gIStep2)) *
                      sd::math::sd_ceil<double, T>(static_cast<double>(hend - hstart) / static_cast<double>(gIStep3)) *
                      sd::math::sd_ceil<double, T>(static_cast<double>(wend - wstart) /
                                                   static_cast<double>(gIStep4));  // Accounts for dilation
                else if (extraParam0 == 1)                                         // Include padding
                  valO /= kProd;

                for (sd::LongType kd = dstart; kd < dend; kd += gIStep2)
                  for (sd::LongType kh = hstart; kh < hend; kh += gIStep3)
                    for (sd::LongType kw = wstart; kw < wend; kw += gIStep4) pgI[kd + kh + kw] += valO;
              }
            }
          }
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
  }
  /*************************************************************************/
  else if (poolingMode == 2) {  // pnorm
    auto func = PRAGMA_THREADS_FOR_2D {
      sd::LongType dstart, hstart, wstart, dend, hend, wend, maxKD, maxKH, maxKW;
      T sum, valO, *pIn, *pgI;

      for (int b = start_x; b < stop_x; b++) {
        for (int c = start_y; c < stop_y; c++) {
          for (int od = 0; od < oD; od++) {
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

                if (dstart < 0) dstart += dD * ((-dstart + dD - 1) / dD);
                if (hstart < 0) hstart += dH * ((-hstart + dH - 1) / dH);
                if (wstart < 0) wstart += dW * ((-wstart + dW - 1) / dW);
                if (dend > iD) dend -= dD * ((dend - iD + dD - 1) / dD);
                if (hend > iH) hend -= dH * ((hend - iH + dH - 1) / dH);
                if (wend > iW) wend -= dW * ((wend - iW + dW - 1) / dW);

                sum = static_cast<T>(0.);
                valO = gO[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4];

                if (sameStrides) {
                  dstart *= iStride2;
                  dend *= iStride2;
                  hstart *= iStride3;
                  hend *= iStride3;
                  wstart *= iStride4;
                  wend *= iStride4;

                  for (sd::LongType kd = dstart; kd < dend; kd += iStep2)
                    for (sd::LongType kh = hstart; kh < hend; kh += iStep3)
                      for (sd::LongType kw = wstart; kw < wend; kw += iStep4)
                        sum += sd::math::sd_pow<T, T, T>(sd::math::sd_abs<T,T>(pIn[kd + kh + kw]), static_cast<T>(extraParam0));

                  valO *= sd::math::sd_pow<T, T, T>(sum, ((T)1.f - extraParam0) / extraParam0);

                  for (sd::LongType kd = dstart; kd < dend; kd += iStep2)
                    for (sd::LongType kh = hstart; kh < hend; kh += iStep3)
                      for (sd::LongType kw = wstart; kw < wend; kw += iStep4)
                        pgI[kd + kh + kw] +=
                            valO *
                            sd::math::sd_pow<T, T, T>(sd::math::sd_abs<T,T>(pIn[kd + kh + kw]), extraParam0 - (T)1.f) *
                            sd::math::sd_sgn<T, T>(pIn[kd + kh + kw]);
                } else {
                  for (sd::LongType kd = dstart; kd < dend; kd += dD)
                    for (sd::LongType kh = hstart; kh < hend; kh += dH)
                      for (sd::LongType kw = wstart; kw < wend; kw += dW)
                        sum += sd::math::sd_pow<T, T, T>(
                            sd::math::sd_abs<T,T>(pIn[kd * iStride2 + kh * iStride3 + kw * iStride4]), static_cast<T>(extraParam0));

                  valO *= sd::math::sd_pow<T, T, T>(sum, ((T)1.f - extraParam0) / extraParam0);

                  for (sd::LongType kd = dstart; kd < dend; kd += dD)
                    for (sd::LongType kh = hstart; kh < hend; kh += dH)
                      for (sd::LongType kw = wstart; kw < wend; kw += dW) {
                        const auto inVal = pIn[kD * iStride2 + kh * iStride3 + kw * iStride4];
                        pgI[kd * gIStride2 + kh * gIStride3 + kw * gIStride4] +=
                            valO * sd::math::sd_pow<T, T, T>(sd::math::sd_abs<T,T>(inVal), static_cast<T>(extraParam0) - 1.f) *
                            sd::math::sd_sgn<T, T>(inVal);
                      }
                }
              }
            }
          }
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1);
  } else {
    sd_printf(
        "ConvolutionUtils::pooling3dBP: pooling mode argument can take three values only: 0, 1, 2, but got %i instead "
        "!\n",
        poolingMode);
    throw "";
  }
}

void ConvolutionUtils::pooling3dBP(sd::graph::Context& block, NDArray& input, NDArray& gradO,
                                   NDArray& gradI, const LongType kD, const LongType kH, const LongType kW, const LongType sD, const LongType sH,
                                   const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD, const LongType dH,
                                   const LongType dW, const int poolingMode, const int extraParam0) {
  BUILD_SINGLE_SELECTOR(
      input.dataType(), pooling3dBP_,
      (block, input, gradO, gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0),
      SD_FLOAT_TYPES);
}
}  // namespace ops
}  // namespace sd
