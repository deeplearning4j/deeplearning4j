/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_in_top_k)

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/top_k.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(in_top_k, 2, 1, true, 0, 1) {
            auto predictions = INPUT_VARIABLE(0);
            auto target = INPUT_VARIABLE(1);

            auto result = OUTPUT_VARIABLE(0);
            REQUIRE_TRUE(block.numI() > 0, 0, "in_top_k: Parameter k is needed to be set");
            REQUIRE_TRUE(predictions->sizeAt(0) == target->lengthOf(), 0, "in_top_k: the number of predictions rows should be equal to target array length, but got %i and %i correspondingly !", predictions->sizeAt(0), target->lengthOf());
            REQUIRE_TRUE(predictions->rankOf() == 2, 0, "in_top_k: The predictions array should have rank 2, but %i given", predictions->rankOf());
            REQUIRE_TRUE(target->rankOf() == 1, 0, "in_top_k: The target should be a vector");

            int k = INT_ARG(0);
            return helpers::inTopKFunctor(block.launchContext(), predictions, target, result, k);
        }

        DECLARE_SHAPE_FN(in_top_k) {
            auto shapeList = SHAPELIST();
            auto in = inputShape->at(1);
            int shapeRank = shape::rank(in);

            auto aShape = ConstantShapeHelper::getInstance().createShapeInfo(sd::DataType::BOOL, shape::order(in), shape::rank(in), shape::shapeOf(in));
            shapeList->push_back(aShape);
            return shapeList;
        }

        DECLARE_TYPES(in_top_k) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_INTS})
                    ->setAllowedOutputTypes(DataType::BOOL);
        }

    }
}

#endif