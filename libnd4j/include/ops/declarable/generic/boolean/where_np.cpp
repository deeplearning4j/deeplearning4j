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
//  @author Adam Gibson
//

#include <op_boilerplate.h>
#include <ops/declarable/headers/boolean.h>

#if NOT_EXCLUDED(OP_where_np)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(where_np, -1, 1, false, 0, 0) {
            auto condition = INPUT_VARIABLE(0);

            if (block.width() == 3) {
                auto x = INPUT_VARIABLE(1);
                auto y = INPUT_VARIABLE(2);

                auto z = OUTPUT_VARIABLE(0);
               int numMatches = 0;
                // if cond matches x/y shape - we have per-element mask
                if (condition->isSameShape(x)) {
                    // FIXME: for perf it might be better to issue memcpy here, and fill only mismatched values from either X or Y
                    if(y->isScalar()) {
                        if (y->isR()) {
                            for (int e = 0; e < condition->lengthOf(); e++) {
                                auto r = condition->e<bool>(e) ? y->e<double>(0)
                                                                              : x->e<double>(e);
                                z->p(e, r);
                            }
                        } else {
                            for (int e = 0; e < condition->lengthOf(); e++) {
                                auto r = condition->e<bool>(e) ? y->e<Nd4jLong>(0)
                                                                              : x->e<Nd4jLong>(e);
                                z->p(e, r);
                            }
                        }
                    }
                    else {
                        if (y->isR()) {
                            for (int e = 0; e < condition->lengthOf(); e++) {
                                if (condition->e<bool>(e)) {
                                    auto r = y->e<double>(numMatches);
                                    z->p(e, r);
                                    numMatches++;
                                } else {
                                    auto r = x->e<double>(e);
                                    z->p(e, r);
                                }
                            }
                        } else {
                            for (int e = 0; e < condition->lengthOf(); e++) {
                                if (condition->e<bool>(e)) {
                                    auto r = y->e<Nd4jLong>(numMatches);
                                    z->p(e, r);
                                    numMatches++;
                                } else {
                                    auto r = x->e<Nd4jLong>(e);
                                    z->p(e, r);
                                }
                            }
                        }
                    }
                }
                else {
                    REQUIRE_TRUE(condition->lengthOf() == x->sizeAt(0), 0, "Condition length should be equal to the dim0 of x/y to act as TAD-mask, but got %d instead", condition->lengthOf());

                    auto dims = ShapeUtils::convertAxisToTadTarget(x->rankOf(), {0});
                    auto tadsX = x->allTensorsAlongDimension(dims);
                    auto tadsY = y->allTensorsAlongDimension(dims);
                    auto tadsZ = z->allTensorsAlongDimension(dims);

                    for (int e = 0; e < tadsX->size(); e++) {
                        if (!condition->e<bool>(e))
                            tadsZ->at(e)->assign(tadsY->at(e));
                        else
                            tadsZ->at(e)->assign(tadsX->at(e));
                    }

                    delete tadsX;
                    delete tadsY;
                    delete tadsZ;
                }
            } else {
                // in this case we return 2D matrix, which basically contains coordinates fo true

                REQUIRE_TRUE(block.width() == 1, 0, "Where op takes either 1 or 3 operands, But got %d operands instead", block.width());

                Nd4jLong width = condition->rankOf();
                nd4j::ops::Where op;
                std::unique_ptr<ResultSet> res(op.execute({condition}, {}, {}));
                REQUIRE_OK(res->status());
                NDArray* whereTrue = res->at(0);
                for (Nd4jLong outNext = 0; outNext < width; ++outNext) {
                    auto output = OUTPUT_VARIABLE(outNext);
                    for (Nd4jLong e = 0; e < output->lengthOf(); ++e) {
                        output->p<Nd4jLong>(e, whereTrue->e<Nd4jLong>(e, outNext));
                        //throw std::runtime_error("Not implemented");
                    }
                }
//                auto result = list.stack();
//                OVERWRITE_RESULT(result);
            }

            return ND4J_STATUS_OK;
        }



        DECLARE_SHAPE_FN(where_np) {
            auto shapes = SHAPELIST();
            Nd4jLong *newShape;
            if (block.width() == 3) {
                auto inShape = inputShape->at(1);
                COPY_SHAPE(inShape, newShape);

                shapes->push_back(newShape);
            } else {
                auto condition = INPUT_VARIABLE(0);

                Nd4jLong numOfTrue = 0LL; //condition->reduceNumber(reduce::CountNonZero).e<Nd4jLong>(0);
                for (Nd4jLong i = 0; i < condition->lengthOf(); ++i)
                    if (condition->e<bool>(i)) numOfTrue++;

                // output shape - a tuple of rank(inShape) 1D tensors with numOfTrue len
                for (Nd4jLong e = 0; e < condition->rankOf(); ++e) {
                    Nd4jLong *newShape;
//                    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);
  //                  shape::shapeVector(numOfTrue, newShape);
                    newShape = ShapeBuilders::createVectorShapeInfo(nd4j::DataType::INT64, numOfTrue, block.workspace());
                    //ArrayOptions::setDataType(newShape, nd4j::DataType::INT64);
                    shapes->push_back(newShape);
                }
            }
            return shapes;
        }
    }
}

#endif