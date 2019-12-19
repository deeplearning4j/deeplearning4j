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

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(rgb_to_grs, 1, 1, false, 0, 0) {
    
    const auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);
    const int rank = input->rankOf();
    const int arg_size = block.getIArguments()->size();
    const int dimC = arg_size > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;
    REQUIRE_TRUE(rank >= 1, 0, "RGBtoGrayScale: Fails to meet the rank requirement: %i >= 1 ", rank);
    REQUIRE_TRUE(rank >= 1, 0, "RGBGrayScale: Fails to meet the rank requirement: %i >= 1 ", rank);
    if (arg_size > 0) {
        REQUIRE_TRUE(dimC >= 0 && dimC < rank, 0, "Index of the Channel dimension out of range: %i not in [%i,%i) ", INT_ARG(0), -rank, rank);
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
    const int rank = input->rankOf();
    REQUIRE_TRUE(rank >= 1, 0, "RGBtoGrayScale: Fails to meet the rank requirement: %i >= 1 ", rank);
    const int dimLast = input->sizeAt(rank - 1);
    REQUIRE_TRUE(dimLast == 3, 0, "RGBtoGrayScale: operation expects 3 channels (R, B, G) in last dimention, but received %i", dimLast);
    auto nShape = input->getShapeInfoAsVector();
    nShape[rank - 1] = 1;
    return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(input->dataType(), input->ordering(), nShape));
}


CONFIGURABLE_OP_IMPL(hsv_to_rgb, 1, 1, false, 0, 0) {

    auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty())
        return Status::OK();

    const int rank = input->rankOf();
    const int arg_size = block.getIArguments()->size();
    const int dimC = arg_size > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;

    REQUIRE_TRUE(rank >= 1, 0, "HSVtoRGB: Fails to meet the rank requirement: %i >= 1 ", rank);
    if (arg_size > 0) {
        REQUIRE_TRUE(dimC >= 0 && dimC < rank, 0, "Index of the Channel dimension out of range: %i not in [%i,%i) ", INT_ARG(0), -rank, rank);
    }
    REQUIRE_TRUE(input->sizeAt(dimC) == 3, 0, "HSVtoRGB: operation expects 3 channels (H, S, V), but got %i instead", input->sizeAt(dimC));

    helpers::transformHsvRgb(block.launchContext(), input, output, dimC);

    return Status::OK();
}

CONFIGURABLE_OP_IMPL(rgb_to_hsv, 1, 1, false, 0, 0) {

    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty())
        return Status::OK();

    const int rank = input->rankOf();
    const int arg_size = block.getIArguments()->size();
    const int dimC = arg_size > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;

    REQUIRE_TRUE(rank >= 1, 0, "RGBtoHSV: Fails to meet the rank requirement: %i >= 1 ", rank);
    if (arg_size > 0) {
        REQUIRE_TRUE(dimC >= 0 && dimC < rank, 0, "Index of the Channel dimension out of range: %i not in [%i,%i) ", INT_ARG(0), -rank, rank);
    }
    REQUIRE_TRUE(input->sizeAt(dimC) == 3, 0, "RGBtoHSV: operation expects 3 channels (H, S, V), but got %i instead", input->sizeAt(dimC));

    helpers::transformRgbHsv(block.launchContext(), input,  output, dimC);

    return Status::OK();
}


DECLARE_TYPES(hsv_to_rgb) {
    getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
        ->setSameMode(true);
}

DECLARE_TYPES(rgb_to_hsv) {
    getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
        ->setSameMode(true);
}


}
}
