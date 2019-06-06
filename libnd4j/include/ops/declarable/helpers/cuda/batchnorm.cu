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
// @author Yurii Shyrma, created on 25.02.2018
//


#include<ops/declarable/helpers/batchnorm.h>
#include <helpers/ShapeUtils.h>
#include <OmpLaunchHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
void batchnorm(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon) {

}

}
}
}

