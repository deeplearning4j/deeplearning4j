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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/helpers.h>

namespace sd    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////
void dilation2d(sd::LaunchContext* context, NDArray *input, NDArray *weights, NDArray *output, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW);

//////////////////////////////////////////////////////////////////////
FORCEINLINE Nd4jStatus outputSize(sd::LaunchContext * context, const int inSize, const int k, const int d, const int s, bool isSameMode, int *outSize, int *padding_before, int *padding_after) {
    if (s <= 0)
        return Status::THROW("Dilation2D: Stride must be > 0");

    if (d < 1)
        return Status::THROW("Dilation2D: Dilation rate must be >= 1");

    int kEff = (k - 1) * d + 1;
    if (isSameMode) {
        *outSize = (inSize + s - 1) / s;
        const int padding_needed = sd::math::nd4j_max<int>(0, (*outSize - 1) * s + kEff -inSize);

        *padding_before = padding_needed / 2;
        *padding_after = padding_needed - *padding_before;
    } else {
        *outSize = (inSize - kEff + s) / s;
        *padding_before = *padding_after = 0;
    }

    if (*outSize < 0)
        return Status::THROW("Dilation2D: outSize has negative value");

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////
FORCEINLINE Nd4jStatus dilation_hw(sd::LaunchContext * context, Nd4jLong const* in, Nd4jLong const* wh, std::vector<int> &strides, std::vector<int> &rates, bool isSameMode, int *sH, int *sW, int *pH, int *pW, int *dH, int *dW, int *oH, int *oW) {
    const int iH = shape::sizeAt(in, 1);
    const int iW = shape::sizeAt(in, 2);
    const int iC = shape::sizeAt(in, 3);

    *sH = strides[1];
    *sW = strides[2];
    *dH = rates[1];
    *dW = rates[2];

    const int kH = shape::sizeAt(wh, 0);
    const int kW = shape::sizeAt(wh, 1);

    const int kHeff = kH + (kH - 1) * (*dH - 1);
    const int kWeff = kW + (kW - 1) * (*dW - 1);

    int padding_after_unusedA, padding_after_unusedB;
    if (outputSize(context, iH, kHeff, 1, *sH, isSameMode, oH, pH, &padding_after_unusedA) != Status::OK())
        return Status::THROW("Dilation2D: bad height");

    if (outputSize(context, iW, kWeff, 1, *sW, isSameMode, oW, pW, &padding_after_unusedA) != Status::OK())
        return Status::THROW("Dilation2D: bad width");

    return Status::OK();
}



}
}
}