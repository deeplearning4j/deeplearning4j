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
            NDArray<T> *x = INPUT_VARIABLE(0);

            bool replace = false;

            auto arguments = block.getIArguments();
            if (block.width() == 2 && arguments->size() == 0) {
                auto axis = INPUT_VARIABLE(1);
                for (int e = 0; e < axis->lengthOf(); e++) {
                    int ax = (int) axis->getScalar(e);
                    if (ax < 0)
                        ax += x->rankOf();

                    arguments->emplace_back(ax);
                }

                replace = true;
            } else if (arguments->size() == 0) {
                for (int e = x->rankOf() - 1; e >= 0; e--)
                    arguments->emplace_back(e);
            }

            // 0D edge case
            if (x->rankOf() == 0) {
                REQUIRE_TRUE(arguments->size() == 1, 0, "Permute: only one axis is allowed for scalar");
                auto output = OUTPUT_VARIABLE(0);
                if (!block.isInplace())
                    output->assign(x);

                return ND4J_STATUS_OK;
            }

            if(block.isInplace()) {		// in-place
                x->permutei(*arguments);
                STORE_RESULT(x);
            } else {	
                if (!replace) {			// not-in-place        
                    NDArray<T>* output = OUTPUT_VARIABLE(0);
                    // nd4j_printv("permute shape", *arguments);
                    auto result = x->permute(*arguments);
                    output->assign(result);
                    STORE_RESULT(output);
                    delete result;
                } else {
                    auto output = x->dup();
                    output->permutei(*arguments);
                    
                    OVERWRITE_RESULT(output);
                }           
            }
        return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(permute) {
            auto shapeList = SHAPELIST();
            auto arguments = block.getIArguments();
            if (shape::rank(inputShape->at(0)) == 0) {
                Nd4jLong *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), Nd4jLong);
                newshape[0] = 0;
                newshape[1] = 0;
                newshape[2] = 1;
                newshape[3] = 99;
                shapeList->push_back(newshape);
            } else if (inputShape->size() == 1 && arguments->size() > 0) {
                auto outputShapeInfo = ShapeUtils<T>::evalPermShapeInfo(arguments->data(), arguments->size(), *INPUT_VARIABLE(0), block.workspace());
                shapeList->push_back(outputShapeInfo);
            } else if (inputShape->size() == 2) {
                // dead end
                Nd4jLong *newshape;
                COPY_SHAPE(inputShape->at(0), newshape);
                shapeList->push_back(newshape);
            } else {
                int rank = shape::rank(inputShape->at(0));
                for (int e = rank - 1; e >= 0; e--)
                    arguments->emplace_back(e);

                auto outputShapeInfo = ShapeUtils<T>::evalPermShapeInfo(arguments->data(), arguments->size(), *INPUT_VARIABLE(0), block.workspace());
                shapeList->push_back(outputShapeInfo);
            }
    
            return shapeList;
        }
    }
}

#endif