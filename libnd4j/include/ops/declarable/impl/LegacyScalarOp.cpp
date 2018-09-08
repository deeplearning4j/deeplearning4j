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

#include <ops/declarable/LegacyScalarOp.h>


namespace nd4j {
    namespace ops {
        LegacyScalarOp::LegacyScalarOp() : LegacyOp::LegacyOp(1) {
            // no-op
        }

        LegacyScalarOp::LegacyScalarOp(int opNum)  : LegacyOp::LegacyOp(1, opNum){
            // no-op
        }

        LegacyOp* LegacyScalarOp::clone() {
            return new LegacyScalarOp(this->_opNum, this->_scalar);
        }

        LegacyScalarOp::LegacyScalarOp(int opNum, T scalar)  : LegacyOp::LegacyOp(1, opNum){
            _scalar = scalar;
        }

        ShapeList *LegacyScalarOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(newShape);
        }

        Nd4jStatus LegacyScalarOp::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);
            T scalar = (T) 0.0f;
            int offset = 0;
            if (block.width() > 1) {
                auto y = INPUT_VARIABLE(1);
                scalar = y->getScalar(0);
            } else if (block.getTArguments()->size() > 0) {
                scalar = T_ARG(0);
                offset++;
            } else {
                scalar = _scalar;
            }

            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            NativeOpExcutioner::execScalar(opNum, x->getBuffer(), x->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), scalar, block.getTArguments()->data() + offset);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
    }
}