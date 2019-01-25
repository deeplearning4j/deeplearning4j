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
// @author Yurii Shyrma, created on 21.09.2018
// @author raver119@gmail.com
//


#include <helpers/TAD.h>
#include<ops/declarable/helpers/ismax.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {

template <typename T>
static void ismax_(const NDArray* input, NDArray* output, const std::vector<int>& dimensions) {

}


void ismax(graph::LaunchContext* context, const NDArray *input, NDArray *output, const std::vector<int>& dimensions) {
    BUILD_SINGLE_SELECTOR(input->dataType(), ismax_, (input, output, dimensions), LIBND4J_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void ismax_, (const NDArray *input, NDArray *output, const std::vector<int>& dimensions), LIBND4J_TYPES);

}
}
}

