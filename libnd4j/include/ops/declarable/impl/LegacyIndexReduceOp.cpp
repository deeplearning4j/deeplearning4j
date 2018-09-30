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
// Created by raver119 on 16.10.2017.
//

#include <ops/declarable/LegacyIndexReduceOp.h>
#include <helpers/ShapeUtils.h>
#include <Status.h>


namespace nd4j {
    namespace ops {
        LegacyIndexReduceOp::LegacyIndexReduceOp() : LegacyOp::LegacyOp(1){
            //
        }

        LegacyIndexReduceOp::LegacyIndexReduceOp(int opNum) : LegacyOp::LegacyOp(1, opNum) {
            //
        }

        LegacyOp* LegacyIndexReduceOp::clone() {
            return new LegacyIndexReduceOp(this->_opNum);
        }

        ShapeList *LegacyIndexReduceOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;
            if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT)) {
                // in this case we just return scalar
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
                newShape[0] = 2;
                newShape[1] = 1;
                newShape[2] = 1;
                newShape[3] = 1;
                newShape[4] = 1;
                newShape[6] = 1;
                newShape[7] = 99;
            } else {
                // in this case we're building proper shape for reduction
                auto array = new NDArray(nullptr, inShape, block.getWorkspace());
                array->triggerAllocationFlag(false, false);

                newShape = ShapeUtils::evalReduceShapeInfo('c', *block.getIArguments(), *array, false, true, block.workspace());

                delete array;
            }

            ArrayOptions::setDataType(newShape, nd4j::DataType::INT64);

            return SHAPELIST(newShape);
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        Nd4jStatus LegacyIndexReduceOp::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            if (z->dataType() != INT64) {
                throw std::runtime_error("IndexReduce operations require output to be INT64");
            }

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            bool allAxes = false;

            if (block.width() == 1) {
                if (block.getIArguments()->size() == 0 ||
                    (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT)) {
                    // scalar
                    NativeOpExcutioner::execIndexReduceScalar(opNum, x->getBuffer(), x->getShapeInfo(),
                                                                         block.getTArguments()->data(), z->getBuffer(), z->getShapeInfo());
                } else {
                    // TAD
                    std::vector<int> dims(*block.getIArguments());
                    for (int e = 0; e < dims.size(); e++)
                        if (dims[e] < 0)
                            dims[e] += x->rankOf();

                    if (dims.size() > 1)
                        std::sort(dims.begin(), dims.end());

                    shape::TAD tad(x->getShapeInfo(), dims.data(), dims.size());
                    tad.createTadOnlyShapeInfo();
                    tad.createOffsets();

                    NativeOpExcutioner::execIndexReduce(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(),
                                                        reinterpret_cast<Nd4jLong *>(z->getBuffer()), z->getShapeInfo(), dims.data(), (int) dims.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);                }
            } else {
                // TF mode
                auto indices = INPUT_VARIABLE(1);
                if (indices->lengthOf() == x->rankOf())
                    allAxes = true;

                std::vector<int> axis(indices->lengthOf());
                for (int e = 0; e < indices->lengthOf(); e++) {
                    // lol otherwise we segfault on macOS
                    int f = indices->e<int>(e);
                    axis[e] = f >= 0 ? f : f += x->rankOf();
                }

                if (allAxes) {

                } else {
                    if (indices->lengthOf() > 1)
                        std::sort(axis.begin(), axis.end());

                    REQUIRE_TRUE(axis.size() > 0, 0, "Some dimensions required for reduction!");

                    shape::TAD tad(x->getShapeInfo(), axis.data(), axis.size());
                    tad.createTadOnlyShapeInfo();
                    tad.createOffsets();

                    NativeOpExcutioner::execIndexReduce(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(),
                                                        reinterpret_cast<Nd4jLong *>(z->getBuffer()), z->getShapeInfo(), axis.data(), (int) axis.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
                }
            }

            STORE_RESULT(*z);

            return Status::OK();
        }
    }
}