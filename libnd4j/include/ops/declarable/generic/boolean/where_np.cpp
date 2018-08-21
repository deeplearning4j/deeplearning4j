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
                        for (int e = 0; e < condition->lengthOf(); e++) {
                            T v = condition->getIndexedScalar(e);
                            T r = v > (T) 0.0f ? y->getIndexedScalar(0) : x->getIndexedScalar(e);
                            z->putIndexedScalar(e, r);
                        }
                    }
                    else {

                        for (int e = 0; e < condition->lengthOf(); e++) {
                            T v = condition->getIndexedScalar(e);
                            if (v > 0.0f) {
                                T r = y->getIndexedScalar(numMatches);
                                z->putIndexedScalar(e, r);
                                numMatches++;
                            }
                            else {
                                T r = x->getIndexedScalar(e);
                                z->putIndexedScalar(e, r);
                            }
                        }
                    }
                }
                else {
                    REQUIRE_TRUE(condition->lengthOf() == x->sizeAt(0), 0, "Condition length should be equal to the dim0 of x/y to act as TAD-mask, but got %d instead", condition->lengthOf());

                    auto dims = ShapeUtils<T>::convertAxisToTadTarget(x->rankOf(), {0});
                    auto tadsX = x->allTensorsAlongDimension(dims);
                    auto tadsY = y->allTensorsAlongDimension(dims);
                    auto tadsZ = z->allTensorsAlongDimension(dims);

                    for (int e = 0; e < tadsX->size(); e++) {
                        T v = condition->getIndexedScalar(e);

                        if (v == (T) 0.0f)
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

                int width = condition->rankOf();

                std::vector<int> dims = ShapeUtils<T>::convertAxisToTadTarget(width, {0});

                NDArrayList<T> list(0, true);
                int cnt = 0;

                Nd4jLong idx[MAX_RANK];
                for (int e = 0; e < condition->lengthOf(); e++) {
                    shape::ind2subC(condition->rankOf(), condition->shapeOf(), e, idx);

                    auto offset = shape::getOffset(0, condition->shapeOf(), condition->stridesOf(), idx, condition->rankOf());
                    T v = condition->buffer()[offset];
                    if (v != (T) 0.0f) {
                        auto array = new NDArray<T>('c', {1, condition->rankOf()});
                        for (int f = 0; f < condition->rankOf(); f++)
                            array->putIndexedScalar(f, (T) idx[f]);

                        list.write(cnt++, array);
                    }
                }

                auto result = list.stack();
                OVERWRITE_RESULT(result);
            }

            return ND4J_STATUS_OK;
        }



        DECLARE_SHAPE_FN(where_np) {
            if (block.width() == 3) {
                auto inShape = inputShape->at(1);
                Nd4jLong *newshape;
                COPY_SHAPE(inShape, newshape);

                return SHAPELIST(newshape);
            } else {
                // FIXME: we can't estimate result here in this case
                auto inShape = inputShape->at(0);
                auto condition = INPUT_VARIABLE(0);

                Nd4jLong numOfTrue = condition->template reduceNumber<simdOps::CountNonZero<T>>();
                // output shape - a tuple of rank(inShape) 1D tensors with numOfTrue len
                ShapeList* shapes = new ShapeList;
                for (Nd4jLong e = 0; e < shape::rank(inShape); ++e) {
                    Nd4jLong *newShape;
                    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);
                    shape::shapeVector(numOfTrue, newShape);
                    shapes->push_back(newShape);
                }
                return shapes;
            }
        }
    }
}

#endif