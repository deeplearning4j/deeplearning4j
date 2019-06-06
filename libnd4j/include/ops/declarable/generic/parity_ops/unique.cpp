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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_unique)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/unique.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(unique, 1, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto values = OUTPUT_VARIABLE(0);
            auto indices = OUTPUT_VARIABLE(1);

            REQUIRE_TRUE(x->dataType() == values->dataType(), 0, "Unique: input and output data types must be the same");

            return helpers::uniqueFunctor(block.launchContext(), x, values, indices,  (NDArray*)nullptr);
        }

        DECLARE_SHAPE_FN(unique) {
            auto in = inputShape->at(0);
            auto source = INPUT_VARIABLE(0);
//            auto shapeList = SHAPELIST(); 
            Nd4jLong* valuesShape;
            Nd4jLong* indicesShape;

            int uniqueCount = helpers::uniqueCount(block.launchContext(), source);

            if (uniqueCount == 0) { // empty value Shape
                valuesShape = ConstantShapeHelper::getInstance()->emptyShapeInfo(source->dataType());
            }
            else {
            // all output shapes are 1D arrays (vectors)
                valuesShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(uniqueCount, ArrayOptions::dataType(in));
            }
            // second output is always LONG
            indicesShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(shape::length(in), nd4j::DataType::INT64);

            //COPY_SHAPE_EX(in, indicesShape, block.getWorkspace());

            return SHAPELIST(valuesShape, indicesShape);

        }

        CUSTOM_OP_IMPL(unique_with_counts, 1, 3, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto values = OUTPUT_VARIABLE(0);
            auto indices = OUTPUT_VARIABLE(1);
            auto counts = OUTPUT_VARIABLE(2);

            return helpers::uniqueFunctor(block.launchContext(), input, values, indices, counts);
        }

        DECLARE_SHAPE_FN(unique_with_counts) {
            auto in = inputShape->at(0);
            auto source = INPUT_VARIABLE(0);

            int uniqueCount = helpers::uniqueCount(block.launchContext(), source);
            // all output shapes are 1D arrays (vectors)
            // all output shapes are 1D arrays (vectors)
            auto valuesShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(uniqueCount, source->dataType());

            // second output is always LONG
            auto indicesShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(source->lengthOf(), nd4j::DataType::INT64);

            // third one as well
            auto countsShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(uniqueCount, nd4j::DataType::INT64);

            return SHAPELIST(valuesShape, indicesShape, countsShape);
        }

        DECLARE_TYPES(unique) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes(0, {ALL_INTS, ALL_FLOATS})
                    ->setAllowedOutputTypes(1, {ALL_INTS});
        }

        DECLARE_TYPES(unique_with_counts) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_INTS, ALL_FLOATS})
                    ->setAllowedOutputTypes(0, {ALL_INTS, ALL_FLOATS})
                    ->setAllowedOutputTypes(1, {ALL_INTS})
                    ->setAllowedOutputTypes(2, {ALL_INTS});
        }

    }
}

#endif