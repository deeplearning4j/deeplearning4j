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
// Created by raver119 on 29/10/17.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_reshape)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// here iArgs is a vector with (optional) negative of order as first element:
// ({-order, dim1, dim2, dim3, ...})
CUSTOM_OP_IMPL(reshape, 1, 1, false, 0, -2) {

    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    //Special case: empty.reshape(<other empty shape>) -> return empty
    if (x->isEmpty()) {
        REQUIRE_TRUE(z->isEmpty(), 0, "Reshape: when input is empty, output must also be empty");
        return Status::OK();    //No op
    }

    REQUIRE_TRUE(x->lengthOf() == z->lengthOf(), 0, "Reshape: lengths before and after reshape should match, but got %i vs %i", x->lengthOf(), z->lengthOf());

    if (Environment::getInstance()->isDebugAndVerbose())
        nd4j_printv("Reshape: new shape", z->getShapeAsVector());

    z->assign(x->reshape(z->ordering(), z->getShapeAsVector()));

    return Status::OK();
}


DECLARE_TYPES(reshape) {
    getOpDescriptor()
            ->setAllowedInputTypes(0, sd::DataType::ANY)
            ->setAllowedInputTypes(1, {ALL_INTS})
            ->setSameMode(true);
}

DECLARE_SHAPE_FN(reshape) {

    const auto x = INPUT_VARIABLE(0);

    std::vector<int> reshapeArgs;
    std::vector<Nd4jLong> shapeNew;
    char orderNew = 'c';

    if (block.width() == 1) {
        reshapeArgs = *block.getIArguments();
        if(!reshapeArgs.empty()) {
            orderNew = (char) -reshapeArgs[0];
            if(orderNew == 'c' || orderNew == 'f')
                reshapeArgs.erase(reshapeArgs.begin());   // remove first element being order in this case
        }
    }
    else {
        reshapeArgs = INPUT_VARIABLE(1)->getBufferAsVector<int>();
        orderNew = block.numI() > 0 ? (char) -INT_ARG(0) : 'c';
    }

    REQUIRE_TRUE(!reshapeArgs.empty() || x->lengthOf() == 1, 0, "Reshape buffer should have at least 1 dimension !");

    Nd4jLong xLen = x->lengthOf();
    if(x->isEmpty()) {
        xLen = 1;
        for (uint i = 0; i < x->rankOf(); ++i)                            // take into account possible empty shapes
            if(x->sizeAt(i) != 0)
                xLen *= x->sizeAt(i);
    }

    for (uint i = 0; i < reshapeArgs.size(); ++i) {

        if (reshapeArgs[i] == -1) {

            uint shapeLength = 1, numOfZeros = 0;

            for(uint j = 0; j < i; ++j)
                if(reshapeArgs[j] != 0)
                    shapeLength *= reshapeArgs[j];
                else
                    ++numOfZeros;

            for(uint j = i + 1; j < reshapeArgs.size(); ++j) {
                REQUIRE_TRUE(reshapeArgs[j] != -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
                if(reshapeArgs[j] != 0)
                    shapeLength *= reshapeArgs[j];
                else
                    ++numOfZeros;
            }

            const auto dim = xLen / shapeLength;

            if(x->isEmpty() && (1 == dim || 0 == numOfZeros))
                shapeNew.push_back(0);
            else
                shapeNew.push_back(dim);
        }
        else
            shapeNew.push_back(reshapeArgs[i]);
    }

    auto len = shape::prodLong(shapeNew.data(), shapeNew.size());
    REQUIRE_TRUE(x->lengthOf() == len, 0, "Reshape: lengths before and after reshape should match, but got %i vs %i", x->lengthOf(), len);

    return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(x->dataType(), orderNew, shapeNew));
}

}
}

#endif