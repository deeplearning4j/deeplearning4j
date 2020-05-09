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

#include <ops/declarable/LegacyTransformSameOp.h>

#include <legacy/NativeOpExecutioner.h>


namespace sd {
    namespace ops {
        LegacyTransformSameOp::LegacyTransformSameOp() : LegacyOp::LegacyOp(1) {
            this->getOpDescriptor()->allowInplace(true);
        }

        LegacyTransformSameOp::LegacyTransformSameOp(int opNum) : LegacyOp::LegacyOp(1, opNum) {
            this->getOpDescriptor()->allowInplace(true);
        }

        LegacyOp* LegacyTransformSameOp::clone() {
            return new LegacyTransformSameOp(this->_opNum);
        }

        Nd4jStatus LegacyTransformSameOp::validateAndExecute(Context &block) {
            auto input = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            NDArray::prepareSpecialUse({z}, {input});

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            ExtraArguments extras(*block.getTArguments());
            PointersManager manager(block.launchContext(), "LegacyTransformSameOp");

            NativeOpExecutioner::execTransformSame(block.launchContext(), opNum, input->buffer(), input->shapeInfo(), input->specialBuffer(), input->specialShapeInfo(),
                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), extras.argumentsAsT(z->dataType()), nullptr, nullptr);

            manager.synchronize();
            STORE_RESULT(*z);

            return Status::OK();
        }

        /**
        * For transform operations, output shape always equals to input shape. With just a few exclusions, like im2col and col2im. 
        * But these ops already have CustomOp implementations.
        *
        */
        ShapeList *LegacyTransformSameOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(CONSTANT(newShape));
        }
    }
}