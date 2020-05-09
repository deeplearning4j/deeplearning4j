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

#include <helpers/ShapeUtils.h>
#include <ops/declarable/LegacyPairwiseTransformBoolOp.h>


namespace sd {
    namespace ops {
        LegacyPairwiseTransformBoolOp::LegacyPairwiseTransformBoolOp() : LegacyOp::LegacyOp(2) {
            // just a no-op
        }

        LegacyPairwiseTransformBoolOp::LegacyPairwiseTransformBoolOp(int opNum) : LegacyOp::LegacyOp(2, opNum) {
            // just a no-op
        }

        LegacyOp* LegacyPairwiseTransformBoolOp::clone() {
            return new LegacyPairwiseTransformBoolOp(this->_opNum);
        }

        Nd4jStatus LegacyPairwiseTransformBoolOp::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            NDArray::prepareSpecialUse({z}, {x, y});

            if (!x->isSameShape(y))
                REQUIRE_TRUE(x->isSameShape(y) || y->isScalar(), 0, "Node_%i: For Pairwise transforms shapes of both operands should be equal but got %s vs %s", block.getNodeId(), ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            ExtraArguments extras(*block.getTArguments());
        PointersManager manager(block.launchContext(), "LegacyPairwiseTransformBoolOp");

            NativeOpExecutioner::execPairwiseTransform(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                    y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                    extras.argumentsAsT(x->dataType()));

            manager.synchronize();
            STORE_RESULT(*z);

            return Status::OK();
        }

        /**
        *   Output shape of PWT operations always the same as input[0] shape, no exclusions.
        */
        ShapeList *LegacyPairwiseTransformBoolOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
            auto inShape = inputShape->at(0);
            return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(inShape, DataType::BOOL)));
        }
    }
}