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
static void pooling3d_(sd::graph::Context& block, NDArray& input, NDArray& output, const LongType kD, const LongType kH,
                       const LongType kW, const LongType sD, const LongType sH, const LongType sW, const LongType pD, const LongType pH, const LongType pW,
                       const LongType dD, const LongType dH, const LongType dW, const LongType poolingMode, const int extraParam0) {
  // input is  [bS, iC, iD, iH, iW]
  // output is [bS, iC, oD, oH, oW]
  T* out = output.bufferAsT<T>();
  T* in = const_cast<NDArray&>(input).bufferAsT<T>();

  const int kDEff = kD + (kD - 1) * (dD - 1);
  const int kHEff = kH + (kH - 1) * (dH - 1);
  const int kWEff = kW + (kW - 1) * (dW - 1);

  const int bS = input.sizeAt(0);
  const int iC = input.sizeAt(1);
  const int iD = input.sizeAt(2);
  const int iH = input.sizeAt(3);
  const int iW = input.sizeAt(4);
  const int oC = output.sizeAt(1);
  const int oD = output.sizeAt(2);
  const int oH = output.sizeAt(3);
  const int oW = output.sizeAt(4);

  sd_debug("MKL-DNN is not used for pooling3d!\n", 0);

  const sd::LongType iStride0 = input.stridesOf()[0];
  const sd::LongType iStride1 = input.stridesOf()[1];
  const sd::LongType iStride2 = input.stridesOf()[2];
  const sd::LongType iStride3 = input.stridesOf()[3];
  const sd::LongType iStride4 = input.stridesOf()[4];
  const sd::LongType oStride0 = output.stridesOf()[0];
  const sd::LongType oStride1 = output.stridesOf()[1];
  const sd::LongType oStride2 = output.stridesOf()[2];
  const sd::LongType oStride3 = output.stridesOf()[3];
  const sd::LongType oStride4 = output.stridesOf()[4];
  const sd::LongType iStep2 = dD * iStride2;
  const sd::LongType iStep3 = dH * iStride3;
  const sd::LongType iStep4 = dW * iStride4;
  const int kProd = kD * kH * kW;

  if (poolingMode == 0) {  // max
    auto func = PRAGMA_THREADS_FOR_3D {
      sd::LongType dstart, hstart, wstart, dend, hend, wend;
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

                if (dstart < 0) dstart += dD * ((-dstart + dD - 1) / dD);
                if (hstart < 0) hstart += dH * ((-hstart + dH - 1) / dH);
                if (wstart < 0) wstart += dW * ((-wstart + dW - 1) / dW);
                if (dend > iD) dend -= dD * ((dend - iD + dD - 1) / dD);
                if (hend > iH) hend -= dH * ((hend - iH + dH - 1) / dH);
                if (wend > iW) wend -= dW * ((wend - iW + dW - 1) / dW);

                dstart *= iStride2;
                dend *= iStride2;
                hstart *= iStride3;
                hend *= iStride3;
                wstart *= iStride4;
                wend *= iStride4;

                sum = -DataTypeUtils::max<T>();

                for (sd::LongType kd = dstart; kd < dend; kd += iStep2)
                  for (sd::LongType kh = hstart; kh < hend; kh += iStep3)
                    for (sd::LongType kw = wstart; kw < wend; kw += iStep4) {
                      T val = pIn[kd + kh + kw];
                      if (val > sum) sum = val;
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
  else if (poolingMode == 1) {  // avg
    auto func = PRAGMA_THREADS_FOR_3D {
      sd::LongType dstart, hstart, wstart, dend, hend, wend;
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

                if (dstart < 0) dstart += dD * ((-dstart + dD - 1) / dD);
                if (hstart < 0) hstart += dH * ((-hstart + dH - 1) / dH);
                if (wstart < 0) wstart += dW * ((-wstart + dW - 1) / dW);
                if (dend > iD) dend -= dD * ((dend - iD + dD - 1) / dD);
                if (hend > iH) hend -= dH * ((hend - iH + dH - 1) / dH);
                if (wend > iW) wend -= dW * ((wend - iW + dW - 1) / dW);

                dstart *= iStride2;
                dend *= iStride2;
                hstart *= iStride3;
                hend *= iStride3;
                wstart *= iStride4;
                wend *= iStride4;

                sum = static_cast<T>(0.);

                for (sd::LongType kd = dstart; kd < dend; kd += iStep2)
                  for (sd::LongType kh = hstart; kh < hend; kh += iStep3)
                    for (sd::LongType kw = wstart; kw < wend; kw += iStep4) sum += pIn[kd + kh + kw];

                if (extraParam0 == 0)  // Exclude padding
                  sum /=
                      sd::math::sd_ceil<double, T>(static_cast<double>(dend - dstart) / static_cast<double>(iStep2)) *
                      sd::math::sd_ceil<double, T>(static_cast<double>(hend - hstart) / static_cast<double>(iStep3)) *
                      sd::math::sd_ceil<double, T>(static_cast<double>(wend - wstart) /
                                                   static_cast<double>(iStep4));  // Accounts for dilation
                else if (extraParam0 == 1)                                        // Include padding
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
  else if (poolingMode == 2) {  // pnorm
    auto func = PRAGMA_THREADS_FOR_3D {
      sd::LongType dstart, hstart, wstart, dend, hend, wend;
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

                if (dstart < 0) dstart += dD * ((-dstart + dD - 1) / dD);
                if (hstart < 0) hstart += dH * ((-hstart + dH - 1) / dH);
                if (wstart < 0) wstart += dW * ((-wstart + dW - 1) / dW);
                if (dend > iD) dend -= dD * ((dend - iD + dD - 1) / dD);
                if (hend > iH) hend -= dH * ((hend - iH + dH - 1) / dH);
                if (wend > iW) wend -= dW * ((wend - iW + dW - 1) / dW);

                dstart *= iStride2;
                dend *= iStride2;
                hstart *= iStride3;
                hend *= iStride3;
                wstart *= iStride4;
                wend *= iStride4;

                sum = static_cast<T>(0.);

                for (sd::LongType kd = dstart; kd < dend; kd += iStep2)
                  for (sd::LongType kh = hstart; kh < hend; kh += iStep3)
                    for (sd::LongType kw = wstart; kw < wend; kw += iStep4)
                      sum += sd::math::sd_pow<T, T, T>(sd::math::sd_abs<T,T>(pIn[kd + kh + kw]), static_cast<T>(extraParam0));

                sum = sd::math::sd_pow<T, T, T>(sum, (T)1.f / extraParam0);

                out[b * oStride0 + c * oStride1 + od * oStride2 + oh * oStride3 + ow * oStride4] = sum;
              }
            }
          }
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, bS, 1, 0, iC, 1, 0, oD, 1);
  } else {
    sd_printf(
        "ConvolutionUtils::pooling3d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead "
        "!\n",
        poolingMode);
    THROW_EXCEPTION("Incorrect poooling3d mode");
  }
}

void ConvolutionUtils::pooling3d(sd::graph::Context& block, NDArray& input, NDArray& output, const LongType kD,
                                 const LongType kH, const LongType kW, const LongType sD, const LongType sH, const LongType sW, const LongType pD,
                                 const LongType pH, const LongType pW, const LongType dD, const LongType dH, const LongType dW,
                                 const int poolingMode, const int extraParam0) {
  BUILD_SINGLE_SELECTOR(
      input.dataType(), pooling3d_,
      (block, input, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0), SD_FLOAT_TYPES);
}

}  // namespace ops
}  // namespace sd
