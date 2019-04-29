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

#include <ops/declarable/LegacyBroadcastBoolOp.h>
#include <helpers/TAD.h>
#include <helpers/ConstantTadHelper.h>
#include <Status.h>


namespace nd4j {
    namespace ops {
        Nd4jStatus LegacyBroadcastBoolOp::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            std::vector<int> dims(*block.getIArguments());
            if (dims.size() > 0)
                std::sort(dims.begin(), dims.end());


            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x->shapeInfo(), dims);

            REQUIRE_TRUE(shape::length(tadPack.primaryShapeInfo()) == y->lengthOf(), 0, "Length of broadcast TAD should be equal to length of Y operand, but got [%i] vs [%i]", (int) shape::length(tadPack.primaryShapeInfo()), (int) y->lengthOf());

            if (x == z)
                NativeOpExcutioner::execBroadcast(opNum, x->buffer(), x->shapeInfo(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), dims.data(), dims.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
            else {
                // this is rare, but possible use case - X and Z might have different shapes/strides/orders. In this case we prepare and pass separate TAD info
                auto tadPackZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(z->shapeInfo(), dims);

                NativeOpExcutioner::execBroadcast(opNum, x->buffer(), x->shapeInfo(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), dims.data(), dims.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets(), tadPackZ.primaryShapeInfo(), tadPackZ.primaryOffsets());
            }

            STORE_RESULT(*z);

            return Status::OK();
        }

        LegacyBroadcastBoolOp::LegacyBroadcastBoolOp() : LegacyOp::LegacyOp(2) {
            //
        }

        LegacyBroadcastBoolOp::LegacyBroadcastBoolOp(int opNum) : LegacyOp::LegacyOp(2, opNum) {
            //
        }

        LegacyOp* LegacyBroadcastBoolOp::clone() {
            return new LegacyBroadcastBoolOp(this->_opNum);
        }

        /**
        *   If external NDArray wasn't specified - the same shape is returned by all broadcast ops.
        */
        ShapeList* LegacyBroadcastBoolOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto inShape = inputShape->at(0);

            // FIXME: remove memcpy
            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);
            ArrayOptions::setDataType(newShape, DataType::BOOL);

            return SHAPELIST(newShape);
        }
    }
}
