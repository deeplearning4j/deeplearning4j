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

#include <ops/declarable/LegacyBroadcastOp.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/axis.h>
#include <helpers/ShapeUtils.h>
#include <helpers/ConstantTadHelper.h>
#include <Status.h>

namespace nd4j {
    namespace ops {
        Nd4jStatus LegacyBroadcastOp::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            std::vector<int> dims(*block.getAxis());
            if (dims.size() == 0 && block.width() > 2) {
                auto axis = INPUT_VARIABLE(2);
                helpers::adjustAxis(x, axis, dims);
                //dims = ShapeUtils::convertAxisToTadTarget(z->rankOf(), dims);
            }
            if (dims.size() > 0)
                std::sort(dims.begin(), dims.end());


            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x->shapeInfo(), dims);
            auto tadLen = shape::length(tadPack.primaryShapeInfo());
            REQUIRE_TRUE(tadLen == y->lengthOf(), 0, "Length of broadcast TAD should be equal to length of Y operand, but got [%i] vs [%i]",tadLen, (int) y->lengthOf());

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

        LegacyBroadcastOp::LegacyBroadcastOp() : LegacyOp::LegacyOp(2) {
            //
        }

        LegacyBroadcastOp::LegacyBroadcastOp(int opNum) : LegacyOp::LegacyOp(2, opNum) {
            //
        }

        LegacyOp* LegacyBroadcastOp::clone() {
            return new LegacyBroadcastOp(this->_opNum);
        }

        /**
        *   If external NDArray wasn't specified - the same shape is returned by all broadcast ops.
        */
        ShapeList* LegacyBroadcastOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto inShape = inputShape->at(0);

            // FIXME: remove memcpy
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), Nd4jLong);
            memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

            return SHAPELIST(newShape);
        }
    }
}
