/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <graph/execution/LogicExecutor.h>
#include <graph/execution/LogicScope.h>
#include <graph/execution/LogicWhile.h>
#include <graph/execution/LogicSwitch.h>
#include <graph/execution/LogicConditional.h>
#include <graph/execution/LogicReturn.h>
#include <graph/execution/LogicExpose.h>
#include <graph/execution/LogicMerge.h>
#include <graph/execution/LogicEnter.h>
#include <graph/execution/LogicExit.h>
#include <graph/execution/LogicLoopCond.h>
#include <graph/execution/LogicNextIteration.h>


namespace sd {
    namespace graph {
        Nd4jStatus LogicExecutor::processNode(Graph *graph, Node *node) {
            switch (node->opNum()) {
                case sd::logic::While:
                    return LogicWhile::processNode(graph, node);
                case sd::logic::Scope:
                    return LogicScope::processNode(graph, node);
                case sd::logic::Conditional:
                    return LogicConditional::processNode(graph, node);
                case sd::logic::Switch:
                    return LogicSwitch::processNode(graph, node);
                case sd::logic::Return:
                    return LogicReturn::processNode(graph, node);
                case sd::logic::Expose:
                    return LogicExpose::processNode(graph, node);
                case sd::logic::Merge:
                    return LogicMerge::processNode(graph, node);
                case sd::logic::LoopCond:
                    return LogicLoopCond::processNode(graph, node);
                case sd::logic::NextIteration:
                    return LogicNextIeration::processNode(graph, node);
                case sd::logic::Exit:
                    return LogicExit::processNode(graph, node);
                case sd::logic::Enter:
                    return LogicEnter::processNode(graph, node);
            }

            if (node->getName() == nullptr) {
                nd4j_printf("Unknown LogicOp used at node [%i]: [%i]\n", node->id(), node->opNum());
            } else {
                nd4j_printf("Unknown LogicOp used at node [%i:<%s>]: [%i]\n", node->id(), node->getName()->c_str(), node->opNum());
            }
            return ND4J_STATUS_BAD_INPUT;
        }
    }
}