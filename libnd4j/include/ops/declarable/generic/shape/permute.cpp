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
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_permute)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {

//////////////////////////////////////////////////////////////////////////
// here iArgs is int vector of ordered set of dimensions to be permuted
        CUSTOM_OP_IMPL(permute, 1, 1, true, 0, -2) {
            auto x = INPUT_VARIABLE(0);

            bool replace = false;

            auto origArgs = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<int>() : *block.getIArguments();
            std::vector<int> arguments({});
            if(origArgs.size() > 0){
                for (int e = 0; e < origArgs.size(); e++) {
                    int ax = origArgs[e];
                    if (ax < 0)
                        ax += x->rankOf();

                    arguments.emplace_back(ax);
                }

                replace = true;
            } else {
                for (int e = x->rankOf() - 1; e >= 0; e--)
                    arguments.emplace_back(e);
            }

            // 0D edge case
            if (x->rankOf() == 0) {
                REQUIRE_TRUE(arguments.size() == 1, 0, "Permute: only one axis is allowed for scalar");
                auto output = OUTPUT_VARIABLE(0);
                if (!block.isInplace())
                    output->assign(x);

                return Status::OK();
            }

            if(block.isInplace()) {		// in-place
                x->permutei(arguments);
                STORE_RESULT(x);
            } else {
                auto output = OUTPUT_VARIABLE(0);
                auto result = x->permute(arguments);
                output->assign(result);
                STORE_RESULT(output);
                delete result;
            }

            return Status::OK();
        }

        DECLARE_TYPES(permute) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(1, {ALL_INTS})
                    ->setSameMode(true);
        }

        DECLARE_SHAPE_FN(permute) {
            auto shapeList = SHAPELIST();
            auto arguments = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<int>() : *block.getIArguments();

            if (shape::rank(inputShape->at(0)) == 0) {
                shapeList->push_back(ConstantShapeHelper::getInstance()->scalarShapeInfo(ArrayOptions::dataType(inputShape->at(0))));
            } else if (inputShape->size() == 1 && !arguments.empty()) {
                shapeList->push_back(ShapeUtils::evalPermShapeInfo(arguments.data(), arguments.size(), *INPUT_VARIABLE(0), block.workspace()));
            } else {
                if(arguments.size() == 0){
                    //Reverse dimensions
                    int rank = shape::rank(inputShape->at(0));
                    for (int e = rank - 1; e >= 0; e--)
                        arguments.emplace_back(e);
                }

                shapeList->push_back(ShapeUtils::evalPermShapeInfo(arguments.data(), arguments.size(), *INPUT_VARIABLE(0), block.workspace()));
            }
    
            return shapeList;
        }
    }
}

#endif