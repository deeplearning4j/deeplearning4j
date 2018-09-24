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
#if NOT_EXCLUDED(OP_where)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/where.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(Where, 1, 1, false, 0, 0) {
            auto condition = INPUT_VARIABLE(0);

            if (block.width() == 3) {
                auto x = INPUT_VARIABLE(1);
                auto y = INPUT_VARIABLE(2);

                auto z = OUTPUT_VARIABLE(0);

                REQUIRE_TRUE(x->isSameShape(y), 0, "X and Y must have equal shapes");

                // if cond matches x/y shape - we have per-element mask
                if (condition->isSameShape(x)) {
                    // FIXME: for perf it might be better to issue memcpy here, and fill only mismatched values from either X or Y
                    for (int e = 0; e < condition->lengthOf(); e++) {
                        if (y->isR()) {
                            auto r = !condition->getIndexedScalar<bool>(e) ? y->getIndexedScalar<double>(e)
                                                                           : x->getIndexedScalar<double>(e);
                            z->putIndexedScalar(e, r);
                        } else {
                            auto r = !condition->getIndexedScalar<bool>(e) ? y->getIndexedScalar<Nd4jLong>(e)
                                                                           : x->getIndexedScalar<Nd4jLong>(e);
                            z->putIndexedScalar(e, r);
                        }
                    }
                } else {
                    REQUIRE_TRUE(condition->lengthOf() == x->sizeAt(0), 0, "Condition length should be equal to the dim0 of x/y to act as TAD-mask, but got %d instead", condition->lengthOf());

                    auto dims = ShapeUtils::convertAxisToTadTarget(x->rankOf(), {0});
                    auto tadsX = x->allTensorsAlongDimension(dims);
                    auto tadsY = y->allTensorsAlongDimension(dims);
                    auto tadsZ = z->allTensorsAlongDimension(dims);

                    for (int e = 0; e < tadsX->size(); e++) {
                        if (!condition->getIndexedScalar<bool>(e)) {
                            tadsZ->at(e)->assign(tadsY->at(e));
                        } else {
                            tadsZ->at(e)->assign(tadsX->at(e));
                        }
                    }

                    delete tadsX;
                    delete tadsY;
                    delete tadsZ;
                }
            } else {
                // in this case we return 2D matrix, which basically contains coordinates fo true

                REQUIRE_TRUE(block.width() == 1, 0, "Where op takes either 1 or 3 operands, But got %d operands instead", block.width());

                int width = condition->rankOf();

                std::vector<int> dims = ShapeUtils::convertAxisToTadTarget(width, {0});

                auto result = helpers::_where(*condition, block.dataType(), block.workspace());
                OVERWRITE_RESULT(result);
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(Where) {
            if (block.width() == 3) {
                auto inShape = inputShape->at(1);
                Nd4jLong *newshape;
                COPY_SHAPE(inShape, newshape);

                return SHAPELIST(newshape);
            } else {
                // FIXME: we can't estimate result here in this case
                // output shape is the 2D tensor num_true x rankOf (inShape)
                auto condition = INPUT_VARIABLE(0);
                auto inShape = inputShape->at(0);
                Nd4jLong numOfTrue = condition->reduceNumber(reduce::CountNonZero, nullptr).getIndexedScalar<Nd4jLong>(0);
                Nd4jLong *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);

                newshape[0] = 2;
                newshape[1] = numOfTrue;
                newshape[2] = shape::rank(inShape);
                newshape[3] = 1;
                newshape[4] = 1;
                newshape[6] = 1;
                newshape[7] = 99;

                ArrayOptions::setDataType(newshape, block.dataType());

                return SHAPELIST(newshape);
            }
        }
    }
}

#endif