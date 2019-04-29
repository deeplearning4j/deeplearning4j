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
// Created by GS <sgazeos@gmail.com> at 3/30/2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/nth_element.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(nth_element, 2, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto n = INPUT_VARIABLE(1);
            bool reverse = false;
            if (block.getIArguments()->size() > 0)
                reverse = (bool)INT_ARG(0);

            auto output = OUTPUT_VARIABLE(0);
            Nd4jLong lastDim = input->sizeAt(-1);
            int nVal = n->e<int>(0);
            REQUIRE_TRUE(nVal < lastDim && nVal >= 0, 0, "nth_element: n should be non-negative and less than last dimension size (%lld), but %i was given.", lastDim, n);
            REQUIRE_TRUE(input->rankOf() > 0, 0, "nth_element: The rank of input array should be at least 1, but %i is given", input->rankOf());            //
            if (output->lengthOf() == input->lengthOf())
                output->assign(input);
            else {
//                if (!input->isVector() && reverse)
//                    n->assign(lastDim - n->e<Nd4jLong>(0) - 1);
                helpers::nthElementFunctor(block.launchContext(), input, n, output, reverse);
            }
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(nth_element) {

            auto in = inputShape->at(0);
            int outRank = shape::rank(in) - 1;
            Nd4jLong *outputShape = nullptr;
            if (outRank > 1) {
                ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);
                outputShape[0] = outRank;
                for (Nd4jLong e = 0; e < outRank; e++)
                outputShape[e + 1] = in[e + 1];

                ShapeUtils::updateStridesAndType(outputShape, in, shape::order(in));
            }
            else if (outRank == 1) {
                outputShape = ShapeBuilders::createVectorShapeInfo(ArrayOptions::dataType(in), shape::sizeAt(in, 0), block.workspace());
            }
            else {
                //outputShape = shape::createScalarShapeInfo();
                outputShape = ShapeBuilders::createScalarShapeInfo(ArrayOptions::dataType(in), block.workspace());
            }
            return SHAPELIST(outputShape);
        }
        DECLARE_TYPES(nth_element) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes(nd4j::DataType::ANY);
        }

    }
}