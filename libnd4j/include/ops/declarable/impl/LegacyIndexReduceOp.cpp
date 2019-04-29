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
#include <helpers/TAD.h>
#include <Status.h>
#include <helpers/ConstantTadHelper.h>


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
            if (block.getAxis()->size() == 0 && block.width() == 1) {
                // in this case we just return scalar
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
                newShape[0] = 2;
                newShape[1] = 1;
                newShape[2] = 1;
                newShape[3] = 1;
                newShape[4] = 1;
                newShape[6] = 1;
                newShape[7] = 99;
            } else if (block.getAxis()->size()){
                // in this case we're building proper shape for reduction
                auto array = INPUT_VARIABLE(0); //new NDArray(nullptr, inShape, block.getWorkspace());
                //array->triggerAllocationFlag(false);

                newShape = ShapeUtils::evalReduceShapeInfo('c', *block.getAxis(), *array, false, true, block.workspace());

                //delete array;
            }
            else {
                bool allAxes = false;
                auto indices = INPUT_VARIABLE(1);
                Nd4jLong rank = shape::rank(inShape);
                if (indices->lengthOf() == rank)
                    allAxes = true;

                std::vector<int> axis(indices->lengthOf());
                for (int e = 0; e < indices->lengthOf(); e++) {
                    // lol otherwise we segfault on macOS
                    int f = indices->e<int>(e);
                    axis[e] = f >= 0 ? f : f += rank;
                }
                if (allAxes){
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
                    auto array = INPUT_VARIABLE(0); //new NDArray(nullptr, inShape, block.getWorkspace());
                    //array->triggerAllocationFlag(false);

                    newShape = ShapeUtils::evalReduceShapeInfo('c', axis, *array, false, true, block.workspace());
                }
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

            NDArray::prepareSpecialUse({z}, {x});

            if (z->dataType() != INT64) {
                throw std::runtime_error("IndexReduce operations require output to be INT64");
            }

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            bool allAxes = false;

            ExtraArguments extras(*block.getTArguments());
            PointersManager manager(block.launchContext(), "LegacyIndexReduceOp");

            if (block.width() == 1) {
                if (block.getAxis()->size() == 0) {
                    // scalar
                    NativeOpExecutioner::execIndexReduceScalar(block.launchContext(), opNum, x->getBuffer(), x->getShapeInfo(),
                                                                         x->getSpecialBuffer(), x->getSpecialShapeInfo(),
                                                                         extras.argumentsAsT(x->dataType()),
                                                                         z->getBuffer(), z->getShapeInfo(),
                                                                         z->getSpecialBuffer(), z->getSpecialShapeInfo());
                } else {
                    // TAD
                    std::vector<int> dims(block.getAxis()->size());
                    for (size_t e = 0; e < dims.size(); e++) {
                        auto axe = block.getAxis()->at(e);
                        dims[e] = axe < 0 ? axe + x->rankOf(): axe;
                    }
                    if (dims.size() > 1)
                        std::sort(dims.begin(), dims.end());

                    auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x->shapeInfo(), dims);

                    auto pDims = (int *) manager.replicatePointer(dims.data(), dims.size() * sizeof(int));
                    auto pTadShape = (Nd4jLong *) manager.replicatePointer(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
                    auto pTadOffsets = (Nd4jLong *) manager.replicatePointer(tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));

                    NativeOpExecutioner::execIndexReduce(block.launchContext(), opNum, x->getBuffer(), x->getShapeInfo(),
                                                        x->getSpecialBuffer(), x->getSpecialShapeInfo(),
                                                        extras.argumentsAsT(x->dataType()),
                                                        reinterpret_cast<Nd4jLong *>(z->getBuffer()), z->getShapeInfo(),
                                                        z->getSpecialBuffer(), z->getSpecialShapeInfo(),
                                                        pDims, (int) dims.size(),
                                                        pTadShape, pTadOffsets);
                }
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
                    NativeOpExecutioner::execIndexReduceScalar(block.launchContext(), opNum, x->getBuffer(), x->getShapeInfo(),
                                                              x->getSpecialBuffer(), x->getSpecialShapeInfo(),
                                                              extras.argumentsAsT(x->dataType()),
                                                              z->getBuffer(), z->getShapeInfo(), z->getSpecialBuffer(),
                                                              z->getSpecialShapeInfo());

                } else {
                    if (indices->lengthOf() > 1)
                        std::sort(axis.begin(), axis.end());

                    REQUIRE_TRUE(axis.size() > 0, 0, "Some dimensions required for reduction!");

                    auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x->shapeInfo(), axis);

                    NativeOpExcutioner::execIndexReduce(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(), reinterpret_cast<Nd4jLong *>(z->getBuffer()), z->getShapeInfo(), axis.data(), (int) axis.size(), tadPack.primaryShapeInfo(), tadPack.primaryOffsets());
                    auto pDims = (int *) manager.replicatePointer(axis.data(), axis.size() * sizeof(int));
                    auto pTadShape = (Nd4jLong *) manager.replicatePointer(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
                    auto pTadOffsets = (Nd4jLong *) manager.replicatePointer(tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));

                    NativeOpExecutioner::execIndexReduce(block.launchContext(), opNum,
                            x->getBuffer(), x->getShapeInfo(), x->getSpecialBuffer(), x->getSpecialShapeInfo(),
                            extras.argumentsAsT(x->dataType()),
                            reinterpret_cast<Nd4jLong *>(z->getBuffer()),
                            z->getShapeInfo(), z->getSpecialBuffer(), z->getSpecialShapeInfo(),
                            pDims, (int) axis.size(), pTadShape, pTadOffsets);
                }
            }

            manager.synchronize();
            STORE_RESULT(*z);

            return Status::OK();
        }
    }
}