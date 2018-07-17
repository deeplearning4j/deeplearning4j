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
#if NOT_EXCLUDED(OP_matrix_determinant)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/lup.h>
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(matrix_determinant, 1, 1, false, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() >=2, 0, "matrix_determinant: The rank of input array should not less than 2, but %i is given", input->rankOf());
            REQUIRE_TRUE(input->sizeAt(-1) == input->sizeAt(-2), 0, "matrix_determinant: The last two dimmensions should be equal, but %i and %i are given", input->sizeAt(-1), input->sizeAt(-2));

            return helpers::determinant(input, output);
        }

        DECLARE_SHAPE_FN(matrix_determinant) {
            auto inShape = inputShape->at(0);

            Nd4jLong* determinantShape;
            int targetRank = shape::rank(inShape) - 2; // last two dimensions will be reduced to scalar

            if (targetRank == 0) { // scalar only
                determinantShape = shape::createScalarShapeInfo();
            }
            else if (targetRank == 1) { // vector 
                ALLOCATE(determinantShape, block.getWorkspace(), shape::shapeInfoLength(targetRank), Nd4jLong);
                shape::shapeVector(shape::sizeAt(inShape, 0), determinantShape);
            }
            else { // only two last dimensions are excluded
                ALLOCATE(determinantShape, block.getWorkspace(), shape::shapeInfoLength(targetRank), Nd4jLong);

                if (shape::order(inShape) == 'c')
                    shape::shapeBuffer(targetRank, shape::shapeOf(inShape), determinantShape);
                else
                    shape::shapeBufferFortran(targetRank, shape::shapeOf(inShape), determinantShape);
            }
            return SHAPELIST(determinantShape);
        }
    }
}

#endif