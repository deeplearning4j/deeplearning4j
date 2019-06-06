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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/image_suppression.h>
//#include <blas/NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void nonMaxSuppressionV2_(NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {

    }

    void nonMaxSuppressionV2(nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), nonMaxSuppressionV2_, (boxes, scales, maxSize, threshold, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void nonMaxSuppressionV2_, (NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output), NUMERIC_TYPES);

}
}
}