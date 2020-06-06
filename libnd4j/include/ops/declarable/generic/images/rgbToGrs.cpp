/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//

#include <ops/declarable/headers/images.h>
#include <ops/declarable/CustomOperations.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(rgb_to_grs, 1, 1, false, 0, 0) {

    const auto input = INPUT_VARIABLE(0);
         auto output = OUTPUT_VARIABLE(0);

    const int inRank = input->rankOf();
    const int argSize = block.getIArguments()->size();
    const int dimC = argSize > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + inRank) : inRank - 1;

    REQUIRE_TRUE(inRank >= 1, 0, "RGBtoGrayScale: Fails to meet the inRank requirement: %i >= 1 ", inRank);
    if (argSize > 0) {
        REQUIRE_TRUE(dimC >= 0 && dimC < inRank, 0, "Index of the Channel dimension out of range: %i not in [%i,%i) ", INT_ARG(0), -inRank, inRank);
    }
    REQUIRE_TRUE(input->sizeAt(dimC) == 3, 0, "RGBGrayScale: operation expects 3 channels (R, G, B) in last dimention, but received %i instead", input->sizeAt(dimC));

    helpers::transformRgbGrs(block.launchContext(), *input, *output, dimC);
    return Status::OK();
}

DECLARE_TYPES(rgb_to_grs) {
    getOpDescriptor()->setAllowedInputTypes( {ALL_INTS, ALL_FLOATS} )
                     ->setSameMode(true);
}

DECLARE_SHAPE_FN(rgb_to_grs) {

    const auto input = INPUT_VARIABLE(0);
    const int inRank = input->rankOf();
    
    const int argSize = block.getIArguments()->size();
    const int dimC = argSize > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + inRank) : inRank - 1;

    REQUIRE_TRUE(inRank >= 1, 0, "RGBtoGrayScale: Fails to meet the inRank requirement: %i >= 1 ", inRank);
    if (argSize > 0) {
        REQUIRE_TRUE(dimC >= 0 && dimC < inRank, 0, "Index of the Channel dimension out of range: %i not in [%i,%i) ", INT_ARG(0), -inRank, inRank);
    }
    REQUIRE_TRUE(input->sizeAt(dimC) == 3, 0, "RGBtoGrayScale: operation expects 3 channels (R, B, G) in last dimention, but received %i", dimC);

    auto nShape = input->getShapeAsVector();
    nShape[dimC] = 1;

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(input->dataType(), input->ordering(), nShape));
}

}
}
