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

#include <ops/declarable/LegacyReduceOp.h>
#include <helpers/TAD.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        LegacyReduceOp<T>::LegacyReduceOp() : LegacyOp<T>::LegacyOp(1) {
            //
        }

        template <typename T>
        LegacyReduceOp<T>::LegacyReduceOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            //this->_opNum = opNum;
        }

        template <typename T>
        LegacyOp<T>* LegacyReduceOp<T>::clone() {
            return new LegacyReduceOp(this->_opNum);
        }

        template <typename T>
        Nd4jStatus LegacyReduceOp<T>::validateAndExecute(Context<T> &block) {
            auto x = INPUT_VARIABLE(0);


            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();
            nd4j_debug("Executing LegacyReduceOp: [%i]\n", opNum);

            bool allAxes = false;

            if (block.width() == 1) {
                auto z = OUTPUT_VARIABLE(0);

                if (block.getIArguments()->size() == x->rankOf())
                    allAxes = true;

                if ((block.getIArguments()->size() == 0) ||
                    (block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT) || allAxes) {
                    // scalar
                    T res = NativeOpExcutioner<T>::execReduceScalar(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data());
                    z->putScalar(0, res);
                } else {
                    // TAD
                    std::vector<int> dims(*block.getIArguments());

                    for (int e = 0; e < dims.size(); e++)
                        if (dims[e] < 0)
                            dims[e] += x->rankOf();

                    std::sort(dims.begin(), dims.end());

                    REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

                    shape::TAD tad(x->getShapeInfo(), dims.data(), dims.size());
                    tad.createTadOnlyShapeInfo();
                    tad.createOffsets();

                    NativeOpExcutioner<T>::execReduce(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(), z->getBuffer(), z->getShapeInfo(), dims.data(), (int) dims.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);
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
                    int f = (int) indices->getScalar(e);
                    axis[e] = f >= 0 ? f : f += x->rankOf();
                }

                if ((block.getIArguments()->size() == 1 && INT_ARG(0) == MAX_INT) || allAxes) {
                    auto z = OUTPUT_VARIABLE(0);

                    auto b = x->getBuffer();
                    auto s = x->shapeInfo();
                    auto e = block.numT() > 0 ? block.getTArguments()->data() : nullptr;

                    //x->printIndexedBuffer("x");

                    // scalar
                    T res = NativeOpExcutioner<T>::execReduceScalar(opNum, b, s, e);
                    z->putScalar(0, res);
                } else {
                    // TAD
                    if (indices->lengthOf() > 1)
                        std::sort(axis.begin(), axis.end());

                    REQUIRE_TRUE(axis.size() > 0, 0, "Some dimensions required for reduction!");

                    shape::TAD tad(x->getShapeInfo(), axis.data(), axis.size());
                    tad.createTadOnlyShapeInfo();
                    tad.createOffsets();

                    auto newShape = ShapeUtils<T>::evalReduceShapeInfo(x->ordering(), axis, x);
                    auto z = new NDArray<T>(newShape, x->getWorkspace());

                    NativeOpExcutioner<T>::execReduce(opNum, x->getBuffer(), x->getShapeInfo(), block.getTArguments()->data(), z->getBuffer(), z->getShapeInfo(), axis.data(), (int) axis.size(), tad.tadOnlyShapeInfo, tad.tadOffsets);

                    RELEASE(newShape, x->getWorkspace());


                    // keepDims processing, for TF compatibility
                    if (block.getIArguments()->size() > 0 && block.getIArguments()->at(0) == 1) {
                        // z->printShapeInfo("z shape before");
                        std::vector<Nd4jLong> newshape(z->getShapeAsVector());
                        for (int e = 0; e < axis.size(); e++) {
                            auto a = axis.at(e);
                            newshape.insert(newshape.begin() + a, 1);
                        }
                        z->reshapei(z->ordering(), newshape);
                        // z->printShapeInfo("z shape after");
                    }

                    OVERWRITE_RESULT(z);
                }
            }

            return ND4J_STATUS_OK;
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        template <typename T>
        ShapeList *LegacyReduceOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            
            Nd4jLong* inShapeInfo = inputShape->at(0);
            
            const int minNumTArgs     = this->getOpDescriptor()->getNumberOfTArgs();
            const int currentNumTArgs = block.getTArguments()->size();
            
            const bool keepDims = currentNumTArgs > minNumTArgs ? static_cast<bool>(T_ARG(currentNumTArgs - 1)) : false;

            Nd4jLong* outShapeInfo = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inShapeInfo), *block.getIArguments(), inShapeInfo, keepDims, false, block.workspace());            

            return SHAPELIST(outShapeInfo);
        }


        template class ND4J_EXPORT LegacyReduceOp<float>;
        template class ND4J_EXPORT LegacyReduceOp<float16>;
        template class ND4J_EXPORT LegacyReduceOp<double>;
    }
}