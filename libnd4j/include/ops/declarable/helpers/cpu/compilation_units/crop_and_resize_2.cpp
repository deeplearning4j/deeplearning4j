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

#include <ops/declarable/helpers/crop_and_resize.h>
#include "../crop_and_resize.hpp"

namespace nd4j {
    namespace ops {
        namespace helpers {
            BUILD_TRIPLE_TEMPLATE(template void cropAndResizeFunctor_, (NDArray const *images, NDArray const *boxes, NDArray const *indices, NDArray const *cropSize, int method, double extrapolationVal, NDArray *crops), NUMERIC_TYPES_2, FLOAT_TYPES, INTEGER_TYPES);
        }
    }
}