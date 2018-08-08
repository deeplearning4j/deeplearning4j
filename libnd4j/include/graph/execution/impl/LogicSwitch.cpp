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
// Created by raver119 on 21.10.17.
//

#include <pointercast.h>
#include <graph/execution/LogicSwitch.h>
#include <GraphExecutioner.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicSwitch<T>::processNode(Graph<T>* graph, Node<T>* node) {
            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();

            Context<T> ctx(node->getContextPrototype(), __variableSpace);

            // this can be either  our format, or compatible format.
            if (graph->hasScope(node->input()->at(0).first)) {
                nd4j_debug("Node_%i: Scoped mode.\n", node->id());
                // first input is Scope, so it's ours
                int scopeConditionIndex = node->input()->at(0).first;
                auto input = ctx.variable(1);

                auto scopeCondition = graph->scopeById(scopeConditionIndex);
                int lastNode = 0;
                for (auto v: *scopeCondition->nodes()) {
                    GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                    lastNode = v->id();
                }

                // now we should take result of the Scope run, and evaluate it
                auto result = __variableSpace->getVariable(lastNode)->getNDArray();
                //result->printBuffer("Result of the last node");


                std::pair<int, int> pair0(node->id(), 0);
                std::pair<int, int> pair1(node->id(), 1);

                if (!__variableSpace->hasVariable(pair0))
                    __variableSpace->putVariable(pair0, new Variable<T>(nullptr, nullptr, node->id(), 0));

                if (!__variableSpace->hasVariable(pair1))
                    __variableSpace->putVariable(pair1, new Variable<T>(nullptr, nullptr, node->id(), 1));

                if (result->getScalar(0) == (T) 0.0f) {
                    __flowPath->markBranch(node->id(), 0);
                    __variableSpace->getVariable(pair0)->setNDArray(input->getNDArray());
                    __variableSpace->getVariable(pair0)->markRemovable(false);
                } else {
                    __flowPath->markBranch(node->id(), 1);
                    __variableSpace->getVariable(pair1)->setNDArray(input->getNDArray());
                    __variableSpace->getVariable(pair1)->markRemovable(false);
                }
            } else {
                // first input is NOT a Scope, so it's compatible format
                nd4j_debug("Node_%i: Compatible mode.\n", node->id());

                auto input = ctx.variable(0)->getNDArray();
                auto boolean = ctx.variable(1)->getNDArray();

                //input->printIndexedBuffer("0");
                //boolean->printIndexedBuffer("1");

                std::pair<int, int> pair0(node->id(), 0);
                std::pair<int, int> pair1(node->id(), 1);

                if (!__variableSpace->hasVariable(pair0))
                    __variableSpace->putVariable(pair0, new Variable<T>(nullptr, nullptr, node->id(), 0));

                if (!__variableSpace->hasVariable(pair1))
                    __variableSpace->putVariable(pair1, new Variable<T>(nullptr, nullptr, node->id(), 1));

                if (boolean->getScalar(0) == (T) 0.0) {
                    // false
                    nd4j_debug("Node_%i: FALSE branch active\n", node->id());
                    __flowPath->markBranch(node->id(), 0);
                    __variableSpace->getVariable(pair0)->setNDArray(input);
                    __variableSpace->getVariable(pair0)->markRemovable(false);
                } else {
                    //true
                    nd4j_debug("Node_%i: TRUE branch active\n", node->id());
                    __flowPath->markBranch(node->id(), 1);
                    __variableSpace->getVariable(pair1)->setNDArray(input);
                    __variableSpace->getVariable(pair1)->markRemovable(false);
                }
            }

            return ND4J_STATUS_OK;
        };

        template class ND4J_EXPORT LogicSwitch<float>;
        template class ND4J_EXPORT LogicSwitch<float16>;
        template class ND4J_EXPORT LogicSwitch<double>;
        template class ND4J_EXPORT LogicSwitch<int>;
        template class ND4J_EXPORT LogicSwitch<Nd4jLong>;
    }
}
