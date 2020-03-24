/*******************************************************************************
 * Copyright (c) 2020 Konduit, K.K.
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
// Created by GS <sgazeos@gmail.com> at 01/28/2020
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstsq)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/lstsq.h>

namespace sd {
    namespace ops {

        CUSTOM_OP_IMPL(lstsq, 2, 1, false, 0, 0) {
            auto a = INPUT_VARIABLE(0);
            auto b = INPUT_VARIABLE(1);
            auto z = OUTPUT_NULLIFIED(0);
            bool fastFlag = true;
            double l2_factor = 0.;
            if (block.numB() > 0) {
                fastFlag = B_ARG(0);
            }
            if (block.numT() > 0) {
                l2_factor = T_ARG(0);
            }
            REQUIRE_TRUE(a->rankOf() >=2, 0, "lstsq: The rank of input left tensor should not be less than 2, but %i is given", a->rankOf());
            REQUIRE_TRUE(b->rankOf() >=2, 0, "lstsq: The rank of input right tensor should not be less than 2, but %i is given", b->rankOf());

//            REQUIRE_TRUE(a->sizeAt(-1) == a->sizeAt(-2), 0, "lstsq: The last two dimmensions should be equal, but %i and %i are given", a->sizeAt(-1), a->sizeAt(-2));
            REQUIRE_TRUE(a->sizeAt(-2) == b->sizeAt(-2), 0, "lstsq: The last dimmension of left part should be equal to prelast of right part, but %i and %i are given", a->sizeAt(-1), b->sizeAt(-2));
            //REQUIRE_TRUE(l2_factor == 0., 0, "lstsq: Implementation of operation is not finished for factor difference from 0.");
            if (a->isEmpty() || b->isEmpty() || z->isEmpty())
                return Status::OK();

            auto res = helpers::leastSquaresSolveFunctor(block.launchContext(), a, b, l2_factor, fastFlag, z);

            return res;
        }

        CUSTOM_OP_IMPL(solve_ls, 2, 1, false, 0, 0) {
            auto a = INPUT_VARIABLE(0);
            auto b = INPUT_VARIABLE(1);
            auto z = OUTPUT_NULLIFIED(0);
            bool fastFlag = true;
            double l2_factor = 0.;
            if (block.numB() > 0) {
                fastFlag = B_ARG(0);
            }
            if (block.numT() > 0) {
                l2_factor = T_ARG(0);
            }
            REQUIRE_TRUE(a->rankOf() >=2, 0, "lstsq: The rank of input left tensor should not be less than 2, but %i is given", a->rankOf());
            REQUIRE_TRUE(b->rankOf() >=2, 0, "lstsq: The rank of input right tensor should not be less than 2, but %i is given", b->rankOf());

//            REQUIRE_TRUE(a->sizeAt(-1) == a->sizeAt(-2), 0, "lstsq: The last two dimmensions should be equal, but %i and %i are given", a->sizeAt(-1), a->sizeAt(-2));
            REQUIRE_TRUE(a->sizeAt(-2) == b->sizeAt(-2), 0, "lstsq: The last dimmension of left part should be equal to prelast of right part, but %i and %i are given", a->sizeAt(-1), b->sizeAt(-2));
            //REQUIRE_TRUE(l2_factor == 0., 0, "lstsq: Implementation of operation is not finished for factor difference from 0.");
            auto res = Status::OK();
            if (a->isEmpty() || b->isEmpty() || z->isEmpty())
                return res;

            res = helpers::leastSquaresSolveFunctor(block.launchContext(), a, b, l2_factor, fastFlag, z);

            return res;
        }

        DECLARE_SYN(MatrixSolveLs, lstsq);

        DECLARE_SHAPE_FN(lstsq) {
            auto in0 = inputShape->at(0);
            auto in1 = inputShape->at(1);
            auto shapeOf = ShapeUtils::shapeAsVector(in1);
            auto rank = shapeOf.size();
            shapeOf[rank - 2] = shape::sizeAt(in0, -1);

            if (shape::isEmpty(in0) || shape::isEmpty(in1)) {
                shapeOf[rank - 1] = 0; // set output shape to empty
            }
            auto resShape = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(in0), shape::order(in1), shapeOf);//ShapeBuilders::copyShapeInfoAndType(in1, in0, true, block.workspace());
            if (shapeOf[rank - 1] == 0) {
                ArrayOptions::setPropertyBit(resShape, ARRAY_EMPTY);
            }
            return SHAPELIST(resShape);
        }

        DECLARE_TYPES(lstsq) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS})
                    ->setSameMode(false);
        }
        DECLARE_SHAPE_FN(solve_ls) {
            auto in0 = inputShape->at(0);
            auto in1 = inputShape->at(1);
            auto shapeOf = ShapeUtils::shapeAsVector(in1);
            auto rank = shapeOf.size();
            shapeOf[rank - 2] = shape::sizeAt(in0, -1);

            if (shape::isEmpty(in0) || shape::isEmpty(in1)) {
                shapeOf[rank - 1] = 0; // set output shape to empty
            }
            auto resShape = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(in0), shape::order(in1), shapeOf);//ShapeBuilders::copyShapeInfoAndType(in1, in0, true, block.workspace());
            if (shapeOf[rank - 1] == 0) {
                ArrayOptions::setPropertyBit(resShape, ARRAY_EMPTY);
            }
            return SHAPELIST(resShape);
        }

        DECLARE_TYPES(solve_ls) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS})
                    ->setSameMode(false);
        }
    }
}

#endif