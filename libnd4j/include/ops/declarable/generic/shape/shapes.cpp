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
#if NOT_EXCLUDED(OP_shapes_of)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(shapes_of, -1, -1, false, 0, 0) {
            for (int e = 0; e < block.width(); e++) {
                auto x = INPUT_VARIABLE(e);
                auto z = OUTPUT_VARIABLE(e);

                for (int i = 0; i < x->rankOf(); i++)
                    z->putIndexedScalar(i, x->sizeAt(i));
            }

            return Status::OK();
        };
        DECLARE_SYN(shape_n, shapes_of);

        DECLARE_SHAPE_FN(shapes_of) {
            auto shapeList = SHAPELIST();

            for (int e = 0; e < inputShape->size(); e++) {
                auto inShape = inputShape->at(e);

                Nd4jLong *newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);
                shape::shapeVector(shape::rank(inShape), newshape);

                shapeList->push_back(newshape);
            }

            return shapeList;
        };
    }
}

#endif