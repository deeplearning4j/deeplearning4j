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

#include <ops/declarable/LegacyTransformAnyOp.h>

#include <NativeOpExcutioner.h>


namespace nd4j {
    namespace ops {
        LegacyTransformAnyOp::LegacyTransformAnyOp() : LegacyOp::LegacyOp(1) {
            // just a no-op
        }

        LegacyTransformAnyOp::LegacyTransformAnyOp(int opNum) : LegacyOp::LegacyOp(1, opNum) {
            // just a no-op
        }

        LegacyOp* LegacyTransformAnyOp::clone() {
            return new LegacyTransformAnyOp(this->_opNum);
        }

        Nd4jStatus LegacyTransformAnyOp::validateAndExecute(Context &block) {
            auto input = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            NativeOpExcutioner::execTransformAny(opNum, input->getBuffer(), input->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data(), nullptr, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        /**
        * For transform operations, output shape always equals to input shape. With just a few exclusions, like im2col and col2im. 
        * But these ops already have CustomOp implementations.
        *
        */
        ShapeList *LegacyTransformAnyOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(newShape);
        }
    }
}