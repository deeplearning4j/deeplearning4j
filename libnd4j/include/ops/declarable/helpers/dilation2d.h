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
SD_LIB_HIDDEN void dilation2d(sd::LaunchContext *context, NDArray *input, NDArray *weights, NDArray *output,
                              const sd::LongType sH, const sd::LongType sW, const sd::LongType pH, const sd::LongType pW, const sd::LongType dH, const sd::LongType dW);

//////////////////////////////////////////////////////////////////////
SD_INLINE sd::Status outputSize(sd::LaunchContext *context, const sd::LongType inSize, const sd::LongType k, const sd::LongType d, const sd::LongType s,
                                bool isSameMode, sd::LongType *outSize, sd::LongType *padding_before, sd::LongType *padding_after) {
  if (s <= 0) return Logger::logKernelFailureMsg("Dilation2D: Stride must be > 0");

  if (d < 1) return Logger::logKernelFailureMsg("Dilation2D: Dilation rate must be >= 1");

  int kEff = (k - 1) * d + 1;
  if (isSameMode) {
    *outSize = (inSize + s - 1) / s;
    const int padding_needed = sd::math::sd_max<sd::LongType>(0, (*outSize - 1) * s + kEff - inSize);

    *padding_before = padding_needed / 2;
    *padding_after = padding_needed - *padding_before;
  } else {
    *outSize = (inSize - kEff + s) / s;
    *padding_before = *padding_after = 0;
  }

  if (*outSize < 0) return Logger::logKernelFailureMsg("Dilation2D: outSize has negative value");

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE sd::Status dilation_hw(sd::LaunchContext *context, sd::LongType const *in, sd::LongType const *wh,
                                 std::vector<sd::LongType> &strides, std::vector<sd::LongType> &rates, bool isSameMode, sd::LongType *sH, sd::LongType *sW,
                                 sd::LongType *pH, sd::LongType *pW, sd::LongType *dH, sd::LongType *dW, sd::LongType *oH, sd::LongType *oW) {
  const sd::LongType iH = shape::sizeAt(in, 1);
  const sd::LongType iW = shape::sizeAt(in, 2);
  const sd::LongType iC = shape::sizeAt(in, 3);

  *sH = strides[1];
  *sW = strides[2];
  *dH = rates[1];
  *dW = rates[2];

  const sd::LongType kH = shape::sizeAt(wh, 0);
  const sd::LongType kW = shape::sizeAt(wh, 1);

  const sd::LongType kHeff = kH + (kH - 1) * (*dH - 1);
  const sd::LongType kWeff = kW + (kW - 1) * (*dW - 1);

  sd::LongType padding_after_unusedA, padding_after_unusedB;
  if (outputSize(context, iH, kHeff, 1, *sH, isSameMode, oH, pH, &padding_after_unusedA) != sd::Status::OK)
    return Logger::logKernelFailureMsg("Dilation2D: bad height");

  if (outputSize(context, iW, kWeff, 1, *sW, isSameMode, oW, pW, &padding_after_unusedA) != sd::Status::OK)
    return Logger::logKernelFailureMsg("Dilation2D: bad width");

  return sd::Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
