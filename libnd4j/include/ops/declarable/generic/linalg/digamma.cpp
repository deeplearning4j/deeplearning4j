/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_digamma)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/gammaMathFunc.h>

namespace sd {
namespace ops  {

CONFIGURABLE_OP_IMPL(digamma, 1, 1, false, 0, 0) {

    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    helpers::diGamma(block.launchContext(), *x, *z);

    return Status::OK();
}

DECLARE_TYPES(digamma) {
    getOpDescriptor()
            ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
            ->setSameMode(true);
}

}
}

#endif