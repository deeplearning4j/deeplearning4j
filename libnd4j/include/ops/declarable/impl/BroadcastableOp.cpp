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
// Created by raver on 6/6/2018.
//

#include <op_boilerplate.h>
#include <pointercast.h>
#include <ops/declarable/BroadcastableOp.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        BroadcastableOp<T>::BroadcastableOp(const char *name, int numTArgs, int numIArgs) : DeclarableCustomOp<T>::DeclarableCustomOp(2, 1, name, false, numTArgs, numIArgs) {
            //
        }

        template <typename T>
        BroadcastableOp<T>::~BroadcastableOp() {
            // no-op
        }

        template <typename T>
        ShapeList *BroadcastableOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            auto shapeList = SHAPELIST();
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);

            if (shape::equalsSoft(x, y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (shape::isScalar(x) && !shape::isScalar(y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(y, newshape);

                shapeList->push_back(newshape);
            } else if (!shape::isScalar(x) && shape::isScalar(y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (ShapeUtils<T>::areShapesBroadcastable(x, y)) {
                Nd4jLong *newshape = nullptr;
                ShapeUtils<T>::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());

                shapeList->push_back(newshape);
            } else {
                // in this case we'll throw exception later
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            }

            return shapeList;
        }

        template class ND4J_EXPORT BroadcastableOp<float>;
        template class ND4J_EXPORT BroadcastableOp<float16>;
        template class ND4J_EXPORT BroadcastableOp<double>;
        template class ND4J_EXPORT BroadcastableOp<int>;
        template class ND4J_EXPORT BroadcastableOp<Nd4jLong>;
    }
}