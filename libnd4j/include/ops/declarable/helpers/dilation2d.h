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

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    void _dilation2d(NDArray<T> *input, NDArray<T> *weights, NDArray<T> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left);

    FORCEINLINE Nd4jStatus _outputSize(int input_size, int filter_size, int dilation_rate, int stride, bool isSameMode, int *output_size, int *padding_before, int *padding_after) {
        if (stride <= 0)
            return Status::THROW("Dilation2D: Stride must be > 0");
    
        if (dilation_rate < 1)
            return Status::THROW("Dilation2D: Dilation rate must be >= 1");

        int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
        if (isSameMode) {
            *output_size = (input_size + stride - 1) / stride;
            const int padding_needed = nd4j::math::nd4j_max<int>(0, (*output_size - 1) * stride + effective_filter_size -input_size);
      
            *padding_before = padding_needed / 2;
            *padding_after = padding_needed - *padding_before;
        } else {
            *output_size = (input_size - effective_filter_size + stride) / stride;
            *padding_before = *padding_after = 0;
        }

        if (*output_size < 0)
            return Status::THROW("Dilation2D: output_size has negative value");
        
        return Status::OK();
    }


    FORCEINLINE Nd4jStatus _dilation_hw(Nd4jLong *in, Nd4jLong *wh, std::vector<int> &strides, std::vector<int> &rates, bool isSameMode, int *stride_rows, int *stride_cols, int *rate_rows, int *rate_cols, int *pad_top, int *pad_left, int *out_rows, int *out_cols) {
        const int input_rows = shape::sizeAt(in, 1);
        const int input_cols = shape::sizeAt(in, 2);
        const int depth = shape::sizeAt(in, 3);

        *stride_rows = strides[1];
        *stride_cols = strides[2];
        *rate_rows = rates[1];
        *rate_cols = rates[2];

        const int filter_rows = shape::sizeAt(wh, 0);
        const int filter_cols = shape::sizeAt(wh, 1);

        const int filter_rows_eff = filter_rows + (filter_rows - 1) * (*rate_rows - 1);
        const int filter_cols_eff = filter_cols + (filter_cols - 1) * (*rate_cols - 1);

        int padding_after_unusedA, padding_after_unusedB;
        if (_outputSize(input_rows, filter_rows_eff, 1, *stride_rows, isSameMode, out_rows, pad_top, &padding_after_unusedA) != Status::OK())
            return Status::THROW("Dilation2D: bad height");

        if (_outputSize(input_cols, filter_cols_eff, 1, *stride_cols, isSameMode, out_cols, pad_left, &padding_after_unusedA) != Status::OK())
            return Status::THROW("Dilation2D: bad width");

        return Status::OK();
    }
}
}
}