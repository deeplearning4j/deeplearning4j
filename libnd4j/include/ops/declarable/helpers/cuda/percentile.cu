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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 17.05.2018
//

#include <ops/declarable/helpers/percentile.h>
#include <NDArrayFactory.h>
#include "ResultSet.h"

namespace nd4j    {
namespace ops     {
namespace helpers {


    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void _percentile(const NDArray& input, NDArray& output, std::vector<int>& axises, const float q, const int interpolation) {

    }

    void percentile(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, std::vector<int>& axises, const float q, const int interpolation) {
        BUILD_SINGLE_SELECTOR(input.dataType(), _percentile, (input, output, axises, q, interpolation), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _percentile, (const NDArray& input, NDArray& output, std::vector<int>& axises, const float q, const int interpolation), LIBND4J_TYPES);

}
}
}