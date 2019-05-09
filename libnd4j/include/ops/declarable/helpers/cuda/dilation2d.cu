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
    template <typename X, typename Y>
    static void dilation2d_(NDArray *input, NDArray *weights, NDArray *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left) {

    };

    void dilation2d(nd4j::LaunchContext * context, NDArray *input, NDArray *weights, NDArray *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), dilation2d_, (input, weights, output, stride_rows, stride_cols, rate_rows, rate_cols, pad_top, pad_left), LIBND4J_TYPES, FLOAT_TYPES);
    }

    BUILD_DOUBLE_TEMPLATE(template void dilation2d_, (NDArray *input, NDArray *weights, NDArray *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left), LIBND4J_TYPES, FLOAT_TYPES);

}
}
}