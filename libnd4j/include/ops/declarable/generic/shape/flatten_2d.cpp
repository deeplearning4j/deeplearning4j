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
// Created by raver119 on 29/10/17.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_flatten2d)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// here iArgs is a vector with (optional) negative of order as first element:
// ({-order, dim1, dim2, dim3, ...})
CUSTOM_OP_IMPL(flatten_2d, 1, 1, false, 0, -2) {

    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    //Special case: empty.reshape(<other empty shape>) -> return empty
    if (x->isEmpty()) {
        REQUIRE_TRUE(z->isEmpty(), 0, "Reshape: when input is empty, output must also be empty");
        return Status::OK();    //No op
    }

    REQUIRE_TRUE(x->lengthOf() == z->lengthOf(), 0, "Reshape: lengths before and after reshape should match, but got %i vs %i", x->lengthOf(), z->lengthOf());

    if (Environment::getInstance().isDebugAndVerbose())
        nd4j_printv("Reshape: new shape", z->getShapeAsVector());

    z->assign(x->reshape(z->ordering(), z->getShapeAsVector()));

    return Status::OK();
}


DECLARE_TYPES(flatten_2d) {
    getOpDescriptor()
            ->setAllowedInputTypes(0, sd::DataType::ANY)
            ->setAllowedInputTypes(1, {ALL_INTS})
            ->setSameMode(true);
}

DECLARE_SHAPE_FN(flatten_2d) {

    const auto x = INPUT_VARIABLE(0);
    const auto shape = x->shapeOf();
     auto axis = INT_ARG(0);
    if(axis < 0) {
        axis += x->rankOf();
    }
    std::vector<int> reshapeArgs;
    std::vector<Nd4jLong> shapeNew;
    auto firstDim = 1;
    auto lastDim = 1;
    for(int i = 0; i < axis; i++) {
        firstDim *= shape[i];
    }

    for(int i = axis; i < x->rankOf(); i++) {
        lastDim *= shape[i];
    }

    shapeNew.push_back(firstDim);
    shapeNew.push_back(lastDim);
    nd4j_printf("Shape %d %d\n",firstDim,lastDim);
    auto len = shape::prodLong(shapeNew.data(), shapeNew.size());
    REQUIRE_TRUE(x->lengthOf() == len, 0, "Reshape: lengths before and after reshape should match, but got %i vs %i", x->lengthOf(), len);

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), x->ordering(), shapeNew));
}
}
}

#endif