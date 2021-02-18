/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_adjust_saturation)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/adjust_saturation.h>
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(adjust_saturation, 1, 1, true, 0, 0) {

    auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // just skip op if input is empty
    if (input->isEmpty())
        return Status::OK();

    const int rank = input->rankOf();
    const int arg_size = block.getIArguments()->size();
    const int dimC = arg_size > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;

    REQUIRE_TRUE(rank >= 3, 0, "ADJUST_SATURATION: op expects rank of input array to be >= 3, but got %i instead", rank);
    if (arg_size > 0) {
        REQUIRE_TRUE(dimC >= 0 && dimC < rank, 0, "Index of the Channel dimension out of range: %i not in [%i,%i) ", INT_ARG(0), -rank, rank);
    }
    REQUIRE_TRUE(input->sizeAt(dimC) == 3, 0, "ADJUST_SATURATION: operation expects image with 3 channels (R, G, B), but got %i instead", input->sizeAt(dimC));
    REQUIRE_TRUE(block.numT() > 0 || block.width() > 1, 0, "ADJUST_SATURATION: scale factor is required !");

    NDArray* factor = nullptr;

    if(block.width() > 1)
        factor = INPUT_VARIABLE(1);
    else {
        factor = new NDArray(output->dataType(), block.launchContext());
        factor->p(0, T_ARG(0));
    }

    helpers::adjustSaturation(block.launchContext(), input, factor, output, dimC);

    if(block.width() == 1)
        delete factor;

    return Status::OK();
}

DECLARE_TYPES(adjust_saturation) {
    getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)
                     ->setSameMode(true);
}



}
}

#endif