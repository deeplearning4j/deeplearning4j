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
//
//

#include <ops/declarable/helpers/s_t_d.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void _spaceTodepth_(NDArray *input, NDArray *output, int block_size, bool isNHWC) {

    }

    void _spaceTodepth(nd4j::LaunchContext * context, NDArray *input, NDArray *output, int block_size, bool isNHWC) {
        BUILD_SINGLE_SELECTOR(input->dataType(), _spaceTodepth_, (input, output, block_size, isNHWC), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _spaceTodepth_, (NDArray *input, NDArray *output, int block_size, bool isNHWC), LIBND4J_TYPES);

}
}
}