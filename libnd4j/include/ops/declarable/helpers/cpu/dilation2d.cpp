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

#include <ops/declarable/helpers/dilation2d.h>
#include <array/DataTypeUtils.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename X, typename Y, typename Z>
    static void __dilation2d(NDArray *input, NDArray *weights, NDArray *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left) {
        const int batch = input->sizeAt(0);
        const int input_rows = input->sizeAt(1);
        const int input_cols = input->sizeAt(2);
        const int depth = input->sizeAt(3);

        const int filter_rows = weights->sizeAt(0);
        const int filter_cols = weights->sizeAt(1);

        const int output_rows = output->sizeAt(1);
        const int output_cols = output->sizeAt(2);

#pragma omp parallel for simd schedule(guided)
        for (int b = 0; b < batch; ++b) {
            for (int h_out = 0; h_out < output_rows; ++h_out) {
                int h_beg = h_out * stride_rows - pad_top;
                for (int w_out = 0; w_out < output_cols; ++w_out) {
                    int w_beg = w_out * stride_cols - pad_left;
                    for (int d = 0; d < depth; ++d) {
                        Z cur_val = -DataTypeUtils::max<Z>();
                        for (int h = 0; h < filter_rows; ++h) {
                            const int h_in = h_beg + h * rate_rows;
                            if (h_in >= 0 && h_in < input_rows) {
                                for (int w = 0; w < filter_cols; ++w) {
                                    const int w_in = w_beg + w * rate_cols;
                                    if (w_in >= 0 && w_in < input_cols) {
                                        const Z val = (*input).getScalar<Z>(b, h_in, w_in, d) + (*weights).getScalar<Z>(h, w, d);
                                        if (val > cur_val) {
                                            cur_val = val;
                                        }
                                    }
                                }
                            }
                        }
                        (*output).putScalar<Z>(b, h_out, w_out, d, cur_val);
                    }
                }
            }
        }
    };

    void _dilation2d(NDArray *input, NDArray *weights, NDArray *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left) {
        auto xType = input->dataType();
        auto yType = weights->dataType();
        auto zType = output->dataType();

        BUILD_TRIPLE_SELECTOR(xType, yType, zType, __dilation2d, (input, weights, output, stride_rows, stride_cols, rate_rows, rate_cols, pad_top, pad_left);, LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }

    BUILD_TRIPLE_TEMPLATE(template void __dilation2d, (NDArray *input, NDArray *weights, NDArray *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left);, LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }
}
}