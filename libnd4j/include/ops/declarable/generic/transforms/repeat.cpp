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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_repeat)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// here iArgs is int vector of repeats at the beginning and last element in iArgs is dimension
CUSTOM_OP_IMPL(repeat, 1, 1, true, 0, -1) {

	auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    std::vector<int> repeats = *block.getIArguments();

    const int axis = repeats.back() < 0 ? repeats.back() + input->rankOf() : repeats.back();

    repeats.pop_back();

    REQUIRE_TRUE(0 <= axis && axis < input->rankOf(), 0, "CUSTOM REPEAT OP: wrong axis argument it should be less then input array rank %i, but got %i instead !", input->rankOf(), axis);

    REQUIRE_TRUE(repeats.size() == 1 || repeats.size() == input->sizeAt(axis), 0, "CUSTOM REPEAT OP: wrong axis argument, size of repeats vector must be 1 or equal to dimension at given axis, but got repeats.size = %i and axis = %i !", repeats.size(), axis);

    input->repeat(axis, repeats, *output);

	return Status::OK();
}

DECLARE_TYPES(repeat) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setSameMode(true);
}

DECLARE_SHAPE_FN(repeat) {

    auto input = INPUT_VARIABLE(0);

    std::vector<int> repeats = *block.getIArguments();

    const int axis = repeats.back() < 0 ? repeats.back() + input->rankOf() : repeats.back();

    repeats.pop_back();

    auto outShape = ShapeUtils::evalRepeatShape(axis, repeats, *input);

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(input->dataType(), input->ordering(), outShape)));

}
}
}

#endif