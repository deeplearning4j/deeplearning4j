/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
//  @author Adam Gibson
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_choose)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/choose.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(choose, -1, 2, false, -1, -1) {

            int mode = INT_ARG(0);
            auto result = OUTPUT_VARIABLE(0);
            auto numResults = OUTPUT_VARIABLE(1);

            if (block.width() > 1) {
                auto arg = INPUT_VARIABLE(0);
                auto comp = INPUT_VARIABLE(1);

                helpers::chooseFunctorArray(arg, comp, mode, result, numResults);

            }//scalar case
            else {
                double scalar = T_ARG(0);
                auto arg = INPUT_VARIABLE(0);
                helpers::chooseFunctorScalar(arg, scalar, mode, result, numResults);
            }


            return Status::OK();
        }

        DECLARE_TYPES(choose) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedOutputTypes(0, {ALL_FLOATS})
                    ->setAllowedOutputTypes(1, {ALL_INTS});
        }

        DECLARE_SHAPE_FN(choose) {
            Nd4jLong *shape;
            int rank;
            if(block.width() > 1) {
                auto first = INPUT_VARIABLE(0);
                auto second = INPUT_VARIABLE(1);
                if(first->lengthOf() > second->lengthOf()) {
                    shape = first->getShapeInfo();
                    rank = first->rankOf();
                }
                else {
                    shape = second->getShapeInfo();
                    rank = second->rankOf();
                }
            }
            else {
                auto first = INPUT_VARIABLE(0);
                shape = first->getShapeInfo();
                rank = first->rankOf();
            }

            Nd4jLong* newShape;
            COPY_SHAPE(shape, newShape);

            auto shapeScalar = ShapeBuilders::createScalarShapeInfo(nd4j::DataType::INT64, block.workspace());

            return SHAPELIST(newShape, shapeScalar);
        }


    }
}

#endif