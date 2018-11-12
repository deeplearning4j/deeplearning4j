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
            int n = (int)INPUT_VARIABLE(1)->getScalar(0);
            bool reverse = false;
            if (block.getIArguments()->size() > 0)
                reverse = (bool)INT_ARG(0);

            auto output = OUTPUT_VARIABLE(0);
            Nd4jLong lastDim = input->sizeAt(-1);
            REQUIRE_TRUE(n < lastDim && n > 0, 0, "nth_element: n should be positive and less than last dimension size, but %i was given.", lastDim);
            REQUIRE_TRUE(input->rankOf() > 0, 0, "nth_element: The rank of input array should be at least 1, but %i is given", input->rankOf());            //
            if (output->lengthOf() == input->lengthOf())
                memcpy(output->buffer(), input->getBuffer(), sizeof(T) * input->lengthOf());
            else {
                if (reverse)
                    n = lastDim - n - 1;
                helpers::nthElementFunctor(input, n, output);
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

                shape::updateStrides(outputShape, shape::order(in));
            }
            else if (outRank == 1) {
                outputShape = ShapeUtils<T>::createVectorShapeInfo(shape::sizeAt(in, 0), block.workspace());
            }
            else {
                //outputShape = shape::createScalarShapeInfo();
                outputShape = ShapeUtils<T>::createScalarShapeInfo(block.workspace());
            }
            return SHAPELIST(outputShape);
        }
    }
}