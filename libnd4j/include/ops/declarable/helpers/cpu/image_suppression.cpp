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

#include <ops/declarable/helpers/axis.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void nonMaxSuppressionV2(NDArray<T>* boxes, NDArray<T>* scales, int maxSize, NDArray<T>* output) {
    }

    template void nonMaxSuppressionV2(NDArray<float>* boxes, NDArray<float>* scales, int maxSize, NDArray<float>* output);
    template void nonMaxSuppressionV2(NDArray<float16>* boxes, NDArray<float16>* scales, int maxSize, NDArray<float16>* output);
    template void nonMaxSuppressionV2(NDArray<double>* boxes, NDArray<double>* scales, int maxSize, NDArray<double>* output);
}
}
}