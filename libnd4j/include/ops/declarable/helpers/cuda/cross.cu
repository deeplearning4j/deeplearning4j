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
// @author GS (sgazeos@gmail.com), created on 10/1/2018
//


#include<ops/declarable/helpers/cross.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void weightedCrossEntropyWithLogitsFunctor_(NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output) {

}

void weightedCrossEntropyWithLogitsFunctor(nd4j::LaunchContext * context, NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output) {
    BUILD_SINGLE_SELECTOR(targets->dataType(), weightedCrossEntropyWithLogitsFunctor_, (targets, input, weights, output), FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void weightedCrossEntropyWithLogitsFunctor_, (NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output), FLOAT_TYPES);

}
}
}