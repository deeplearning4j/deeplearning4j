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
#if NOT_EXCLUDED(OP_select)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(select, 3, 1, false, 0, 0) {
            auto cond = INPUT_VARIABLE(0);
            auto x = INPUT_VARIABLE(1);
            auto y = INPUT_VARIABLE(2);

            REQUIRE_TRUE(x->isSameShape(y), 0, "Select: X and Y shape should be equal");
            if (x->isScalar()) {
                REQUIRE_TRUE(cond->isScalar(), 0,
                             "Select: Condition should gave either equal shape to X/Y first dimension or to be scalar");

                auto z = OUTPUT_VARIABLE(0);

                if (y->isR()) {
                    auto v = !cond->e<bool>(0)? y->e<double>(0) : x->e<double>(0);
                    z->p(0, v);
                } else {
                    auto v = !cond->e<bool>(0)? y->e<Nd4jLong>(0) : x->e<Nd4jLong>(0);
                    z->p(0, v);
                }
            } else {
                bool same = cond->isSameShape(x);
                REQUIRE_TRUE(cond->isScalar() || cond->lengthOf() == x->sizeAt(0) || same, 0, "Select: Condition should gave either equal shape to X/Y first dimension or to be scalar");
                if (same) {
                    auto z = OUTPUT_VARIABLE(0);

                    for (int e = 0; e < cond->lengthOf(); e++) {
                        if (y->isR()) {
                            auto r = !cond->e<bool>(e) ? y->e<double>(e) : x->e<double>(e);
                            z->p(e, r);
                        } else {
                            auto r = !cond->e<bool>(e) ? y->e<Nd4jLong>(e) : x->e<Nd4jLong>(e);
                            z->p(e, r);
                        }
                    }
                } else {
                    REQUIRE_TRUE(cond->lengthOf() == x->sizeAt(0), 0, "Condition length should be equal to the dim0 of x/y to act as TAD-mask, but got %d instead", cond->lengthOf());

                    auto z = OUTPUT_VARIABLE(0);

                    auto dims = ShapeUtils::convertAxisToTadTarget(x->rankOf(), {0});
                    auto tadsX = x->allTensorsAlongDimension(dims);
                    auto tadsY = y->allTensorsAlongDimension(dims);
                    auto tadsZ = z->allTensorsAlongDimension(dims);

                    for (int e = 0; e < tadsX->size(); e++) {
                        if (!cond->e<bool>(e)) {
                            tadsZ->at(e)->assign(tadsY->at(e));
                        } else {
                            tadsZ->at(e)->assign(tadsX->at(e));
                        }
                    }

                    delete tadsX;
                    delete tadsY;
                    delete tadsZ;
                }
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(select) {
            auto inShape = inputShape->at(1);

            Nd4jLong *newshape;
            COPY_SHAPE(inShape, newshape);

            return SHAPELIST(newshape);
        }
    }
}

#endif