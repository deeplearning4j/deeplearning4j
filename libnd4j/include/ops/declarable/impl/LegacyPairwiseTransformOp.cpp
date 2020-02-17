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
#include <ops/declarable/LegacyPairwiseTransformOp.h>


namespace nd4j {
    namespace ops {
        LegacyPairwiseTransformOp::LegacyPairwiseTransformOp() : LegacyOp::LegacyOp(2) {
            this->getOpDescriptor()->allowInplace(true);
        }

        LegacyPairwiseTransformOp::LegacyPairwiseTransformOp(int opNum) : LegacyOp::LegacyOp(2, opNum) {
            this->getOpDescriptor()->allowInplace(true);
        }

        LegacyOp* LegacyPairwiseTransformOp::clone() {
            return new LegacyPairwiseTransformOp(this->_opNum);
        }

        Nd4jStatus LegacyPairwiseTransformOp::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            NDArray::prepareSpecialUse({z}, {x, y});

            if (!x->isSameShape(y)) {
                std::string sx = ShapeUtils::shapeAsString(x);
                std::string sy = ShapeUtils::shapeAsString(y);
                REQUIRE_TRUE(x->isSameShape(y) || y->isScalar(), 0, "Node_%i: For Pairwise transforms shapes of both operands should be equal but got %s vs %s", block.getNodeId(), sx.c_str(), sy.c_str());
            }

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            ExtraArguments extras(*block.getTArguments());
            PointersManager manager(block.launchContext(), "LegacyPairwiseTransformOp");

            NativeOpExecutioner::execPairwiseTransform(block.launchContext(), opNum, x->getBuffer(), x->getShapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                    y->getBuffer(), y->getShapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                    z->getBuffer(), z->getShapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                    extras.argumentsAsT(z->dataType()));

            manager.synchronize();
            STORE_RESULT(*z);

            return Status::OK();
        }

        /**
        *   Output shape of PWT operations always the same as input[0] shape, no exclusions.
        */
        ShapeList *LegacyPairwiseTransformOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(CONSTANT(newShape));
        }
    }
}