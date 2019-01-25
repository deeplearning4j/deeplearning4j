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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.08.2018
//

#include <ops/declarable/helpers/histogramFixedWidth.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void histogramFixedWidth_(const NDArray& input, const NDArray& range, NDArray& output) {

    }

    void histogramFixedWidth(graph::LaunchContext* context, const NDArray& input, const NDArray& range, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), histogramFixedWidth_, (input, range, output), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void histogramFixedWidth_, (const NDArray& input, const NDArray& range, NDArray& output), LIBND4J_TYPES);

}
}
}