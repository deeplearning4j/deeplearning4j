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
#if NOT_EXCLUDED(OP_transpose)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(transpose, 1, 1, false, 0, 0) {

    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    //Special case: empty.reshape(<other empty shape>) -> return empty
    if (x->isEmpty()) {
        REQUIRE_TRUE(z->isEmpty(), 0, "TRANSPOSE OP: when input is empty, output must also be empty");
        return Status::OK();    //No op
    }

    if (block.width() == 1 && block.getIArguments()->size() == 0) {
        z->assign(x->transpose());
        return Status::OK();
    }

    std::vector<int> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<int>() : *block.getIArguments();

    z->assign(x->permute(permutationVector));

    return Status::OK();
}

DECLARE_TYPES(transpose) {
    getOpDescriptor()
            ->setAllowedInputTypes(nd4j::DataType::ANY)
            ->setSameMode(true);
}

DECLARE_SHAPE_FN(transpose) {

    auto x = INPUT_VARIABLE(0);

    if (block.width() == 1 && block.getIArguments()->size() == 0)
        return SHAPELIST(ShapeUtils::evalTranspShapeInfo(*x, block.workspace(), true));

    std::vector<int> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<int>() : *block.getIArguments();

    auto outputShapeInfo = ShapeUtils::evalPermShapeInfo(permutationVector.data(), x->rankOf(), *x, block.workspace(), true);

    return SHAPELIST(outputShapeInfo);
}

}
}

#endif