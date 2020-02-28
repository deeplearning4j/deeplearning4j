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
// Created by GS <sgazeos@gmail.com> at 01/22/2020
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_solve)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/solve.h>
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(solve, 2, 1, false, 0, 0) {
            auto a = INPUT_VARIABLE(0);
            auto b = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            bool useAdjoint = false;

            if (block.numB() > 0) {
                useAdjoint = B_ARG(0);
            }
            REQUIRE_TRUE(shape::shapeEquals(a->rankOf() - 2, a->shapeInfo(), b->rankOf() - 2, b->shapeInfo()), 0, "solve: Input shapes should be alike.");
            REQUIRE_TRUE(a->rankOf() >=2, 0, "solve: The rank of input left tensor should not be less than 2, but %i is given", a->rankOf());
            REQUIRE_TRUE(b->rankOf() >=2, 0, "solve: The rank of input right tensor should not be less than 2, but %i is given", b->rankOf());

            REQUIRE_TRUE(a->sizeAt(-1) == a->sizeAt(-2), 0, "solve: The last two dimmensions should be equal, but %i and %i are given", a->sizeAt(-1), a->sizeAt(-2));
            REQUIRE_TRUE(a->sizeAt(-1) == b->sizeAt(-2), 0, "solve: The last dimmension of left part should be equal to prelast of right part, but %i and %i are given", a->sizeAt(-1), b->sizeAt(-2));
            if (a->isEmpty() || b->isEmpty() || z->isEmpty())
                return Status::OK();

            auto input = a;
            if (useAdjoint) {
                auto adjointA = a->ulike();
                helpers::adjointMatrix(block.launchContext(), a, &adjointA);
                input = new NDArray(adjointA); //.detach();
            };

            auto res = helpers::solveFunctor(block.launchContext(), input, b, useAdjoint, z);
            if (input != a)
                delete input;

            return Status::OK();
        }
        
        DECLARE_SHAPE_FN(solve) {
            auto in0 = inputShape->at(1);
            auto in1 = inputShape->at(1);
            auto luShape = ShapeBuilders::copyShapeInfoAndType(in1, in0, true, block.workspace());

            return SHAPELIST(CONSTANT(luShape));
        }

        DECLARE_TYPES(solve) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS})
                    ->setSameMode(false);
        }
    }
}

#endif