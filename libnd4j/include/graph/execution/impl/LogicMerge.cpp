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
// Created by raver119 on 30.01.18.
//

#include <graph/execution/LogicMerge.h>
#include <Status.h>

namespace nd4j {
    namespace graph {
        template<typename T>
        Nd4jStatus LogicMerge<T>::processNode(Graph<T> *graph, Node<T> *node) {
            // at merge node only one of inputs exist if that's just switch and other node isn't LogicNextItration
            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();

            // merge MUST have 2 inputs
            auto inputAddr0 = node->input()->at(0);
            auto inputAddr1 = node->input()->at(1);

            bool isWhile = false;

            // now we want to check if second input is NextIteration
            if (graph->hasNode(inputAddr1.first)) {
                auto secondNode = graph->nodeById(inputAddr1.first);

                // checking for NextIteration
                if (secondNode->opType() == OpType_LOGIC && secondNode->opNum() == 80L) {
                    isWhile = true;

                    // notifying NextIteration node for rewind index
                    secondNode->setRewindLayer(node->getLayer());
                    secondNode->setRewindNode(node->id());
                }

            }

            // FIXME: we don't need this check. Just last input should survive, IF it exists
            if (isWhile){

                if (node->getFrameId() >= 0)
                    __flowPath->markFrameActive(node->getFrameId(), true);

                bool hasVar = __variableSpace->hasVariable(inputAddr1);
                if ( hasVar && __flowPath->wasExecuted(inputAddr1.first)) {
                    nd4j_debug("Node_%i: propagating second input\n", node->id());
                    auto var = __variableSpace->getVariable(inputAddr1);

                    Variable<T> *lvar = nullptr;
                    if (__variableSpace->hasVariable(node->id(), 0))
                        lvar = __variableSpace->getVariable(node->id(), 0);
                    else
                        lvar = new Variable<T>(nullptr, node->getName()->c_str(), node->id(), 0);

//                    if (lvar->hasNDArray())
//                        delete lvar->getNDArray();

                    auto array = var->getNDArray();

                    //array->printIndexedBuffer("propagated");

                    lvar->setNDArray(array);
                    lvar->markReadOnly(true);

                    __flowPath->markExecuted(inputAddr1.first, false);


                } else {
                    nd4j_debug("Node_%i: propagating first input\n", node->id());
                    auto var = __variableSpace->getVariable(inputAddr0);

                    Variable<T> *lvar = nullptr;
                    if (__variableSpace->hasVariable(node->id(), 0))
                        lvar = __variableSpace->getVariable(node->id(), 0);
                    else
                        lvar = new Variable<T>(nullptr, node->getName()->c_str(), node->id(), 0);

//                    if (lvar->hasNDArray())
//                        delete lvar->getNDArray();

                    auto array = var->getNDArray();
                    lvar->setNDArray(array);
                    lvar->markReadOnly(true);


                }
            } else {

                // basically, first non-null variable is our target
                for (int e = 0; e < node->input()->size(); e++) {
                    auto inputAddr = node->input()->at(e);

                    if (__variableSpace->hasVariable(inputAddr)) {
                        auto var = __variableSpace->getVariable(inputAddr);
                        if (!var->hasNDArray())
                            continue;

                        Variable<T> *lvar = nullptr;
                        if (__variableSpace->hasVariable(node->id(), 0))
                            lvar = __variableSpace->getVariable(node->id(), 0);
                        else
                            lvar = new Variable<T>(nullptr, node->getName()->c_str(), node->id(), 0);

                        if (lvar->hasNDArray())
                            delete lvar->getNDArray();

                        auto array = var->getNDArray();
                        lvar->setNDArray(array);
                        lvar->markReadOnly(true);
                        //lvar->markExternal(false);h

                        break;
                    }
                }
            }

            return Status::OK();
        }


        template class ND4J_EXPORT LogicMerge<float>;
        template class ND4J_EXPORT LogicMerge<float16>;
        template class ND4J_EXPORT LogicMerge<double>;
        template class ND4J_EXPORT LogicMerge<int>;
        template class ND4J_EXPORT LogicMerge<Nd4jLong>;
    }
}
