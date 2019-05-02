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
// Created by GS <sgazeos@gmail.com> at 2/26/2018
//

#include <op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/lup.h>

#if NOT_EXCLUDED(OP_matrix_determinant)
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(matrix_determinant, 1, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() >=2, 0, "matrix_determinant: The rank of input array should not less than 2, but %i is given", input->rankOf());
            REQUIRE_TRUE(input->sizeAt(-1) == input->sizeAt(-2), 0, "matrix_determinant: The last two dimmensions should be equal, but %i and %i are given", input->sizeAt(-1), input->sizeAt(-2));

            return helpers::determinant(block.launchContext(), input, output);
        }

        DECLARE_SHAPE_FN(matrix_determinant) {
            auto inShape = inputShape->at(0);

            Nd4jLong* determinantShape;
            int targetRank = shape::rank(inShape) - 2; // last two dimensions will be reduced to scalar

            if (targetRank == 0) { // scalar only
                determinantShape = ConstantShapeHelper::getInstance()->scalarShapeInfo(ArrayOptions::dataType(inShape));
            }
            else if (targetRank == 1) { // vector 
                determinantShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(shape::sizeAt(inShape, 0), ArrayOptions::dataType(inShape));
            }
            else { // only two last dimensions are excluded                
                determinantShape = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), targetRank, shape::shapeOf(inShape));
            }
            return SHAPELIST(determinantShape);
        }

        DECLARE_TYPES(matrix_determinant) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif

#if NOT_EXCLUDED(OP_log_matrix_determinant)
namespace nd4j {
    namespace ops {
        DECLARE_TYPES(log_matrix_determinant) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

        CUSTOM_OP_IMPL(log_matrix_determinant, 1, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() >=2, 0, "log_matrix_determinant: The rank of input array should not less than 2, but %i is given", input->rankOf());
            REQUIRE_TRUE(input->sizeAt(-1) == input->sizeAt(-2), 0, "log_matrix_determinant: The last two dimmensions should be equal, but %i and %i are given", input->sizeAt(-1), input->sizeAt(-2));

            return helpers::log_abs_determinant(block.launchContext(), input, output);
        }

        DECLARE_SHAPE_FN(log_matrix_determinant) {
            auto inShape = inputShape->at(0);

            Nd4jLong* determinantShape;
            int targetRank = shape::rank(inShape) - 2; // last two dimensions will be reduced to scalar

            if (targetRank == 0) { // scalar only
                determinantShape = ConstantShapeHelper::getInstance()->scalarShapeInfo(ArrayOptions::dataType(inShape));
            }
            else if (targetRank == 1) { // vector 
                determinantShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(shape::sizeAt(inShape, 0), ArrayOptions::dataType(inShape));
            }
            else { // only two last dimensions are excluded
                determinantShape = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), targetRank, shape::shapeOf(inShape));
            }
            return SHAPELIST(determinantShape);
        }
    }
}
#endif

#if NOT_EXCLUDED(OP_logdet)
namespace nd4j {
    namespace ops {
        DECLARE_TYPES(logdet) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

        CUSTOM_OP_IMPL(logdet, 1, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() >=2, 0, "logdet: The rank of input array should not less than 2, but %i is given", input->rankOf());
            REQUIRE_TRUE(input->sizeAt(-1) == input->sizeAt(-2), 0, "logdet: The last two dimmensions should be equal, but %i and %i are given", input->sizeAt(-1), input->sizeAt(-2));
            REQUIRE_TRUE(helpers::checkCholeskyInput(block.launchContext(), input), 0, "logdet: The input tensor should be positive-defined hermitian.");

            return helpers::logdetFunctor(block.launchContext(), input, output);
        }

        DECLARE_SHAPE_FN(logdet) {
            auto inShape = inputShape->at(0);

            Nd4jLong* determinantShape;
            int targetRank = shape::rank(inShape) - 2; // last two dimensions will be reduced to scalar

            if (targetRank == 0) { // scalar only
                determinantShape = ConstantShapeHelper::getInstance()->scalarShapeInfo(ArrayOptions::dataType(inShape));
            }
            else if (targetRank == 1) { // vector 
                determinantShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(shape::sizeAt(inShape, 0), ArrayOptions::dataType(inShape));
            }
            else { // only two last dimensions are excluded
                determinantShape = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), targetRank, shape::shapeOf(inShape));
            }
            return SHAPELIST(determinantShape);
        }
    }
}
#endif
