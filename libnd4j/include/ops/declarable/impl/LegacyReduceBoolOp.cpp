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

#include <ops/declarable/LegacyReduceBoolOp.h>
#include <helpers/TAD.h>
#include <helpers/ShapeUtils.h>
#include <Status.h>

namespace nd4j {
    namespace ops {
        LegacyReduceBoolOp::LegacyReduceBoolOp() : LegacyOp::LegacyOp(1) {
            //
        }

        LegacyReduceBoolOp::LegacyReduceBoolOp(int opNum) : LegacyOp::LegacyOp(1, opNum) {
            //this->_opNum = opNum;
        }

        LegacyOp* LegacyReduceBoolOp::clone() {
            return new LegacyReduceBoolOp(this->_opNum);
        }

        Nd4jStatus LegacyReduceBoolOp::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);


            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();
            nd4j_debug("Executing LegacyReduceFloatOp: [%i]\n", opNum);

            auto axis = *block.getAxis();

            bool allAxes = false;

            ExtraArguments extras(*block.getTArguments());

            if (block.width() == 1) {
                auto z = OUTPUT_VARIABLE(0);

                if (axis.size() == x->rankOf())
                    allAxes = true;

                if ((axis.empty()) ||
                    (axis.size() == 1 && axis[0] == MAX_INT) || allAxes) {
                    // scalar
                    NativeOpExecutioner::execReduceBoolScalar(nullptr, opNum, x->getBuffer(), x->getShapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                            extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
                } else {
                    // TAD
                    std::vector<int> dims(axis);

                    for (int e = 0; e < dims.size(); e++)
                        if (dims[e] < 0)
                            dims[e] += x->rankOf();

                    std::sort(dims.begin(), dims.end());

                    REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

                    shape::TAD tad(x->getShapeInfo(), dims.data(), dims.size());
                    tad.createTadOnlyShapeInfo();
                    tad.createOffsets();

                    CudaManager manager(block.launchContext());
                    auto pDims = manager.replicatePointer(dims.data(), dims.size() * sizeof(int));
                    auto pTadShape = manager.replicatePointer(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
                    auto pTadOffsets = manager.replicatePointer(tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));

                    NativeOpExecutioner::execReduceBool(nullptr, opNum, x->getBuffer(), x->getShapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                            extras.argumentsAsT(x->dataType()),
                            z->getBuffer(), z->getShapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                            reinterpret_cast<int *>(pDims), (int) dims.size(), reinterpret_cast<Nd4jLong *>(pTadShape), reinterpret_cast<Nd4jLong *>(pTadOffsets));

                    manager.syncStream("LegacyReduceBoolOp");
                }

                STORE_RESULT(*z);
            } else {
                auto indices = INPUT_VARIABLE(1);
                if (indices->lengthOf() == x->rankOf())
                    allAxes = true;

                //indices->printIndexedBuffer("indices");

                std::vector<int> axis(indices->lengthOf());
                for (int e = 0; e < indices->lengthOf(); e++) {
                    // lol otherwise we segfault on macOS
                    int f = indices->e<int>(e);
                    axis[e] = f >= 0 ? f : f += x->rankOf();
                }

                if ((block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT) || allAxes) {
                    auto z = OUTPUT_VARIABLE(0);

                    ExtraArguments extras(*block.getTArguments());
                    CudaManager manager(block.launchContext());

                    // scalar
                    NativeOpExecutioner::execReduceBoolScalar(block.launchContext(), opNum, x->getBuffer(), x->getShapeInfo(), x->specialBuffer(), x->specialShapeInfo(), extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());

                    manager.syncStream("LegacyReduceBoolOp");
                } else {
                    // TAD
                    if (indices->lengthOf() > 1)
                        std::sort(axis.begin(), axis.end());

                    REQUIRE_TRUE(axis.size() > 0, 0, "Some dimensions required for reduction!");

                    shape::TAD tad(x->getShapeInfo(), axis.data(), axis.size());
                    tad.createTadOnlyShapeInfo();
                    tad.createOffsets();

                    auto z = OUTPUT_VARIABLE(0);

                    ExtraArguments extras(*block.getTArguments());
                    CudaManager manager(block.launchContext());
                    auto pDims = (int *) manager.replicatePointer(axis.data(), axis.size() * sizeof(int));
                    auto pTadShape = (Nd4jLong *) manager.replicatePointer(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
                    auto pTadOffsets = (Nd4jLong *) manager.replicatePointer(tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));

                    NativeOpExecutioner::execReduceBool(block.launchContext(), opNum, x->getBuffer(), x->getShapeInfo(), x->specialBuffer(), x->specialShapeInfo(), extras.argumentsAsT(x->dataType()),
                            z->getBuffer(), z->getShapeInfo(), z->specialBuffer(), z->specialShapeInfo(), pDims, (int) axis.size(), pTadShape, pTadOffsets);

                    manager.syncStream("LegacyReduceBoolOp");
                }
            }

            return Status::OK();
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        ShapeList *LegacyReduceBoolOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;

            bool allAxes = false;

            auto keepDims = block.numB() > 0 ? B_ARG(0) : false;
            auto newFormat = block.numB() > 1 ? B_ARG(1) : true;

            auto axis = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<int>() : *block.getAxis();

            if (axis.size() == shape::rank(inShape))
                allAxes = true;

            // in this case we're building proper shape for reduction
            auto array = new NDArray(nullptr, inShape, block.getVariableSpace()->launchContext());
            array->triggerAllocationFlag(false, false);
            newShape = ShapeUtils::evalReduceShapeInfo(shape::order(inShape), axis, *array, keepDims, !newFormat, block.workspace());
            ArrayOptions::setDataType(newShape, DataType::BOOL);
            delete array;

            return SHAPELIST(newShape);
        }
    }
}