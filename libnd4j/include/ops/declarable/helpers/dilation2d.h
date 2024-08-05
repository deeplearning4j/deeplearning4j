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
//  @author raver119@gmail.com
//
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////
SD_LIB_HIDDEN void dilation2d(LaunchContext *context, NDArray *input, NDArray *weights, NDArray *output,
                              const LongType sH, const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW);

//////////////////////////////////////////////////////////////////////
SD_INLINE Status outputSize(LaunchContext *context, const LongType inSize, const LongType k, const LongType d, const LongType s,
                                bool isSameMode, LongType *outSize, LongType *padding_before,
                            LongType *padding_after) {
  if (s <= 0) return Logger::logKernelFailureMsg("Dilation2D: Stride must be > 0");

  if (d < 1) return Logger::logKernelFailureMsg("Dilation2D: Dilation rate must be >= 1");

  int kEff = (k - 1) * d + 1;
  if (isSameMode) {
    *outSize = (inSize + s - 1) / s;
    const int padding_needed = sd::math::sd_max<LongType>(0, (*outSize - 1) * s + kEff - inSize);

    *padding_before = padding_needed / 2;
    *padding_after = padding_needed - *padding_before;
  } else {
    *outSize = (inSize - kEff + s) / s;
    *padding_before = *padding_after = 0;
  }

  if (*outSize < 0) return Logger::logKernelFailureMsg("Dilation2D: outSize has negative value");

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE Status dilation_hw(LaunchContext *context, LongType const *in, LongType const *wh,
                                 std::vector<LongType> &strides, std::vector<LongType> &rates, bool isSameMode,
                             LongType *sH, LongType *sW, LongType *pH, LongType *pW, LongType *dH, LongType *dW,
                             LongType *oH, LongType *oW) {
  const LongType iH = shape::sizeAt(in, static_cast<LongType>(1));
  const LongType iW = shape::sizeAt(in, static_cast<LongType>(2));
  const LongType iC = shape::sizeAt(in, static_cast<LongType>(3));

  *sH = strides[1];
  *sW = strides[2];
  *dH = rates[1];
  *dW = rates[2];

  const LongType kH = shape::sizeAt(wh, static_cast<LongType>(0));
  const LongType kW = shape::sizeAt(wh, static_cast<LongType>(1));

  const LongType kHeff = kH + (kH - 1) * (*dH - 1);
  const LongType kWeff = kW + (kW - 1) * (*dW - 1);

  LongType padding_after_unusedA, padding_after_unusedB;
  if (outputSize(context, iH, kHeff, 1, *sH, isSameMode, oH, pH, &padding_after_unusedA) != Status::OK)
    return Logger::logKernelFailureMsg("Dilation2D: bad height");

  if (outputSize(context, iW, kWeff, 1, *sW, isSameMode, oW, pW, &padding_after_unusedA) != Status::OK)
    return Logger::logKernelFailureMsg("Dilation2D: bad width");

  return Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
