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
        BroadcastableOp::BroadcastableOp(const char *name, int numTArgs, int numIArgs) : DeclarableCustomOp::DeclarableCustomOp(2, 1, name, false, numTArgs, numIArgs) {
            //
        }

        BroadcastableOp::~BroadcastableOp() {
            // no-op
        }

        ShapeList *BroadcastableOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto shapeList = SHAPELIST();
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);
            auto outputs = _descriptor->getOutputTypesForOutput(0);
            nd4j::DataType dtype = block.dataType(0);
            if (block.dataType(0) != nd4j::DataType::BOOL && !(outputs.size() == 1 && outputs[0] == nd4j::DataType::BOOL)) {
                if (Environment::getInstance()->isExperimentalBuild()) {
                    if (shape::length(y) > shape::length(x)) {
                        dtype = DataTypeUtils::pickPairwiseResultType(y, x);
                    } else {
                        dtype = DataTypeUtils::pickPairwiseResultType(x, y);
                    }
                } else {
                    dtype = ArrayOptions::dataType(x);
                }
            } else
                dtype = nd4j::DataType::BOOL;

            if(shape::isEmpty(x) || shape::isEmpty(y)) {
                //Edge case: broadcasting with empty array gives empty array output (behaviour to match TF for import cases)
                Nd4jLong* empty = ShapeBuilders::emptyShapeInfo(dtype, block.getWorkspace());
				shapeList->push_back(empty);
			} else if (shape::isScalar(x) && shape::isScalar(y)) {
                if (shape::rank(x) >= shape::rank(y)) {
                    Nd4jLong *newshape;
                    COPY_SHAPE(x, newshape);

                    ArrayOptions::setDataType(newshape, dtype);
                    shapeList->push_back(newshape);
                } else {
                    Nd4jLong *newshape;
                    COPY_SHAPE(y, newshape);

                    ArrayOptions::setDataType(newshape, dtype);
                    shapeList->push_back(newshape);
                }
            } else if (shape::equalsSoft(x, y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                ArrayOptions::setDataType(newshape, dtype);
                shapeList->push_back(newshape);
            } else if (shape::isScalar(x) && !shape::isScalar(y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(y, newshape);

                ArrayOptions::setDataType(newshape, dtype);
                shapeList->push_back(newshape);
            } else if (!shape::isScalar(x) && shape::isScalar(y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                ArrayOptions::setDataType(newshape, dtype);
                shapeList->push_back(newshape);
            } else if (ShapeUtils::areShapesBroadcastable(x, y)) {
                Nd4jLong *newshape = nullptr;
                ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());

                ArrayOptions::setDataType(newshape, dtype);
                shapeList->push_back(newshape);
            } else {
                // in this case we'll throw exception later
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                ArrayOptions::setDataType(newshape, dtype);
                shapeList->push_back(newshape);
            }

            return shapeList;
        }
    }
}