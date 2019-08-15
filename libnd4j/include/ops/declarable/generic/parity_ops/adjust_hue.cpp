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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_adjust_hue)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/adjust_hue.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {


CONFIGURABLE_OP_IMPL(adjust_hue, 1, 1, true, 1, -2) {

    auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const int rank     = input->rankOf();
    const int dimC     = block.getIArguments()->size() > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;
    const double delta = T_ARG(0);

    REQUIRE_TRUE(rank >= 3, 0, "ADJUST_HUE: op expects rank of input array to be >= 3, but got %i instead", rank);
    REQUIRE_TRUE(input->sizeAt(dimC) == 3, 0, "ADJUST_HUE: operation expects image with 3 channels (R, G, B), but got %i instead", input->sizeAt(dimC));
    REQUIRE_TRUE(-1. <= delta && delta <= 1., 0, "ADJUST_HUE: parameter delta must be within [-1, 1] interval, but got %f instead", delta);

    NDArray deltaScalarArr = NDArrayFactory::create<double>(delta, block.launchContext());

    helpers::adjustHue(block.launchContext(), input, &deltaScalarArr, output, dimC);

    return Status::OK();
}

DECLARE_TYPES(adjust_hue) {
    getOpDescriptor()->setAllowedInputTypes(nd4j::DataType::ANY)
                     ->setSameMode(true);
}




}
}

#endif