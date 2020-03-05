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
// Created by raver119 on 01/11/17.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_onehot)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/one_hot.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(onehot, 1, 1, false, -2, -2) {
            auto input = INPUT_VARIABLE(0);

            // FIXME: double?
            double on(1.0f); // T_ARG(0);
            double off(0.0f); //T_ARG(1);

            auto depth = -1; //INT_ARG(0);
            auto axis = -1; //INT_ARG(1);

            if (block.numI() > 0)
                axis = INT_ARG(0);

            if (block.numI() > 1) {
                depth = INT_ARG(1);
            } else if (block.width() > 1) {
                depth = INPUT_VARIABLE(1)->e<int>(0);
            }

            REQUIRE_TRUE(depth > 0, 0, "OneHot: depth must be positive value");


            if (block.width() > 2) {
                on = INPUT_VARIABLE(2)->e<double>(0);

                if (block.width() > 3)
                    off = INPUT_VARIABLE(3)->e<double>(0);
            } else if (block.numT() > 0) {
                on = T_ARG(0);

                if (block.numT() > 1)
                    off = T_ARG(1);
            }

            auto output = OUTPUT_VARIABLE(0);

            if (axis < 0)
                axis = output->rankOf() + axis;

            helpers::onehot(block.launchContext(), input, output, axis, depth, on, off);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(onehot) {
            auto inShape = inputShape->at(0);

            sd::DataType dtype = block.numD() > 0 ? D_ARG(0) : sd::DataType::FLOAT32;

            int depth = -1;
            Nd4jLong axis = -1;

            if (block.numI() > 0)
                axis = INT_ARG(0);

             if (block.numI() > 1) {
                depth = INT_ARG(1);
            } else if (block.width() > 1) {
                depth = INPUT_VARIABLE(1)->e<int>(0);
            }

            REQUIRE_TRUE(depth > 0, 0, "OneHot: depth must be positive value");

            Nd4jLong *newShape;
            int rank = shape::rank(inShape);

            if (axis < 0)
                axis = rank + 1 + axis;

            std::vector<Nd4jLong> shape;
            for (int e = 0; e < rank; e++)
                shape.push_back(shape::shapeOf(inShape)[e]);

            shape.insert(shape.begin() + axis, depth);
            newShape = ConstantShapeHelper::getInstance()->createShapeInfo(dtype, 'c', rank + 1, shape.data());

            return SHAPELIST(newShape);
        }

        DECLARE_TYPES(onehot) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS});
        }
    }
}

#endif