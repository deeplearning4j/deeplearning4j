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
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicConditional.h>
#include <GraphExecutioner.h>
#include <graph/execution/LogicReturn.h>
#include <Status.h>


namespace nd4j {
    namespace graph {
        Nd4jStatus LogicConditional::processNode(Graph *graph, Node *node) {
            auto __variableSpace = graph->getVariableSpace();

            auto size = node->input()->size();

            // propagating inputs (optional)
            for (int e = 0; e < size - 3; e++) {
                std::pair<int, int> pair(node->id(), e);
                if (!__variableSpace->hasVariable(pair)) {
                    __variableSpace->putVariable(pair, new Variable(nullptr, nullptr, node->id(), e));
                }

                auto va = node->input()->at(e);

                auto inputVar = __variableSpace->getVariable(va);

                auto innerVar = __variableSpace->getVariable(pair);
                if (innerVar->hasNDArray()) {
                    // TODO: ???
                } else {
                    // FIXME: in some cases it's possible to have no NDArray
                    if (inputVar->hasNDArray())
                        innerVar->setNDArray(inputVar->getNDArray()->dup());
                }
            }


            int scopeConditionIndex = node->input()->at(size - 3).first;
            int scopeFalseIndex = node->input()->at(size - 2).first;
            int scopeTrueIndex = node->input()->at(size - 1).first;

            auto scopeCondition = graph->scopeById(scopeConditionIndex);
            int lastNode = 0;
            for (auto v: *scopeCondition->nodes()) {
                GraphExecutioner::executeFlatNode(graph, v, __variableSpace);
                lastNode = v->id();
            }

            // now we should take result of the Scope run, and evaluate it
            //nd4j_debug("", "");
            auto result = __variableSpace->getVariable(lastNode)->getNDArray();
            //result->printBuffer("Result of the last node:");

            bool isReturn = false;

            // now we're executing one of the scopes, depending on condition evaluation
            if (result->getScalar<int>(0) ==  0) {
                auto scopeFalse = graph->scopeById(scopeFalseIndex);
                lastNode = 0;
                int nodes = scopeFalse->nodes()->size();
                for (int e = 0; e < nodes - 1; e++) {
                    auto v = scopeFalse->nodes()->at(e);
                    GraphExecutioner::executeFlatNode(graph, v, __variableSpace);
                    lastNode = v->id();
                }

                // last node is either return or just last op
                auto *node = scopeFalse->nodes()->at(nodes -1);
                if (node->opType() == OpType_LOGIC && node->opNum() == 40) {
                    isReturn = true;
                    LogicReturn::processNode(graph, node);
                } else {
                    GraphExecutioner::executeFlatNode(graph, node, __variableSpace);
                    lastNode = node->id();
                }
            } else {
                auto scopeTrue = graph->scopeById(scopeTrueIndex);
                lastNode = 0;
                int nodes = scopeTrue->nodes()->size();
                for (int e = 0; e < nodes - 1; e++) {
                    auto v = scopeTrue->nodes()->at(e);
                    GraphExecutioner::executeFlatNode(graph, v, __variableSpace);
                    lastNode = v->id();
                }

                // last node is either return or just last op
                auto node = scopeTrue->nodes()->at(nodes -1);
                if (node->opType() == OpType_LOGIC && node->opNum() == 40) {
                    isReturn = true;
                    LogicReturn::processNode(graph, node);
                } else {
                    GraphExecutioner::executeFlatNode(graph, node, __variableSpace);
                    lastNode = node->id();
                }
            }

            // now fetch and transfer variables to Conditional node
            // but only if return wasn't called at the end of scope
            if (!isReturn) {
                for (int e = 0; e < 65536; e++) {
                    std::pair<int, int> pair(lastNode, e);
                    std::pair<int, int> pairNew(node->id(), e);
                    if (__variableSpace->hasVariable(pair)) {
                        auto array = __variableSpace->getVariable(pair)->getNDArray();
                        auto newVar = new Variable(array);
                        newVar->setId(lastNode, e);
                        newVar->markRemovable(false);

                        __variableSpace->putVariable(pairNew, newVar);
                    } else
                        break;
                }
            }

            return nd4j::Status::OK();
        }
    }
}