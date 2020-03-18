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
#include <helpers/TAD.h>
#include <helpers/ConstantTadHelper.h>
#include <array/DataTypeUtils.h>

namespace sd {
    namespace ops {
        Nd4jStatus LegacyReduce3Op::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            NDArray::prepareSpecialUse({z}, {x, y});

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            nd4j_debug("Executing LegacyReduce3Op: [%i]\n", opNum);

            ExtraArguments extras(*block.getTArguments());
            PointersManager manager(block.launchContext(), "LegacyReduce3Op");

            if (x->isSameShape(y) && (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == sd::DataTypeUtils::max<int>()))) {
                // reduce3 to scalar
                NativeOpExecutioner::execReduce3Scalar(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                        extras.argumentsAsT(z->dataType()),
                        y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
            } else {
                std::vector<int> dims(*block.getAxis());
                for (int e = 0; e < dims.size(); e++)
                    if (dims[e] < 0)
                        dims[e] += x->rankOf();

                auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(x->getShapeInfo(), dims);
                auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(z->getShapeInfo(), dims);

                REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions requuired for reduction!");

                auto xTadShape = Environment::getInstance()->isCPU() ? packX.primaryShapeInfo() : packX.specialShapeInfo(); //(Nd4jLong *) manager.replicatePointer(tadX.tadOnlyShapeInfo, shape::shapeInfoByteLength(tadX.tadOnlyShapeInfo));
                auto xTadOffsets = Environment::getInstance()->isCPU() ? packX.primaryOffsets() : packX.specialOffsets(); //(Nd4jLong *) manager.replicatePointer(tadX.tadOffsets, tadX.numTads * sizeof(Nd4jLong));

                auto yTadShape = Environment::getInstance()->isCPU() ? packZ.primaryShapeInfo() : packZ.specialOffsets(); //(Nd4jLong *) manager.replicatePointer(tadY.tadOnlyShapeInfo, shape::shapeInfoByteLength(tadY.tadOnlyShapeInfo));
                auto yTadOffsets = Environment::getInstance()->isCPU() ? packZ.primaryOffsets() : packZ.specialOffsets(); //(Nd4jLong *) manager.replicatePointer(tadY.tadOffsets, tadY.numTads * sizeof(Nd4jLong));

                NativeOpExecutioner::execReduce3(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                        extras.argumentsAsT(z->dataType()),
                        y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                        dims.data(), dims.size(), xTadShape, xTadOffsets, yTadShape, yTadOffsets);
            }

            manager.synchronize();
            STORE_RESULT(*z);

            return Status::OK();
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
        ShapeList *LegacyReduce3Op::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
            auto xShape = inputShape->at(0);
            auto yShape = inputShape->at(1);

            Nd4jLong *zShape = nullptr;

            if (shape::equalsSoft(xShape, yShape) && (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == sd::DataTypeUtils::max<int>()))) {
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
                auto array = new NDArray(nullptr, xShape, block.launchContext());

                xShape = ShapeUtils::evalReduceShapeInfo('c', *block.getIArguments(), *array, false, true);

                delete array;
            }

            return SHAPELIST(zShape);
        }
    }
}