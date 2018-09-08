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
// Created by raver119 on 17.10.2017.
//

#include <ops/declarable/LegacyReduce3Op.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        Nd4jStatus LegacyReduce3Op::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            nd4j_debug("Executing LegacyReduce3Op: [%i]\n", opNum);

            if (x->isSameShape(y) && (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT))) {
                // reduce3 to scalar
                T scalar = NativeOpExcutioner::execReduce3Scalar(opNum, x->buffer(), x->shapeInfo(), block.getTArguments()->data(), y->buffer(), y->shapeInfo());
                z->putScalar(0, scalar);
            } else {
                std::vector<int> dims(*block.getIArguments());
                for (int e = 0; e < dims.size(); e++)
                    if (dims[e] < 0)
                        dims[e] += x->rankOf();

                std::sort(dims.begin(), dims.end());

                REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions requuired for reduction!");

                NativeOpExcutioner::execReduce3(opNum, x->buffer(), x->shapeInfo(), block.getTArguments()->data(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), dims.data(), dims.size());
            }


            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        LegacyReduce3Op::LegacyReduce3Op() : LegacyOp::LegacyOp(2) {
            //
        }

        LegacyReduce3Op::LegacyReduce3Op(int opNum) : LegacyOp::LegacyOp(2, opNum) {
            //
        }

        LegacyOp* LegacyReduce3Op::clone() {
            return new LegacyReduce3Op(this->_opNum);
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        ShapeList *LegacyReduce3Op::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto xShape = inputShape->at(0);
            auto yShape = inputShape->at(1);

            Nd4jLong *zShape = nullptr;

            if (shape::equalsSoft(xShape, yShape) && (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT))) {
                // reduce3 to scalar case
                ALLOCATE(zShape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
                zShape[0] = 2;
                zShape[1] = 1;
                zShape[2] = 1;
                zShape[3] = 1;
                zShape[4] = 1;
                zShape[5] = 0;
                zShape[6] = 1;
                zShape[7] = 99;
            } else {
                auto array = new NDArray(nullptr, xShape, block.getWorkspace());
                array->triggerAllocationFlag(false, false);

                xShape = ShapeUtils::evalReduceShapeInfo('c', *block.getIArguments(), *array, false, true);

                delete array;
            }

            return SHAPELIST(zShape);
        }
    }
}