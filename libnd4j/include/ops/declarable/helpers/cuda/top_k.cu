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

#include <ops/declarable/helpers/top_k.h>
#include <ops/declarable/headers/parity_ops.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static int topKFunctor_(graph::LaunchContext* context, NDArray* input, NDArray* values, NDArray* indeces, int k, bool needSort) {
        return Status::OK();
    }
// ----------------------------------------------------------------------------------------------- //

    template <typename T>
    static int inTopKFunctor_(graph::LaunchContext* context, NDArray* input, NDArray* target, NDArray* result, int k) {
        return Status::OK();
    }

    int topKFunctor(graph::LaunchContext* context, NDArray* input, NDArray* values, NDArray* indeces, int k, bool needSort) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return topKFunctor_, (input, values, indeces, k, needSort), NUMERIC_TYPES);
    }

    int inTopKFunctor(graph::LaunchContext* context, NDArray* input, NDArray* target, NDArray* result, int k) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return inTopKFunctor_, (input, target, result, k), NUMERIC_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int topKFunctor_, (graph::LaunchContext* context, NDArray* input, NDArray* values, NDArray* indeces, int k, bool needSort), NUMERIC_TYPES);
    BUILD_SINGLE_TEMPLATE(template int inTopKFunctor_, (graph::LaunchContext* context, NDArray* input, NDArray* target, NDArray* result, int k), NUMERIC_TYPES);
}
}
}