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
//  @author raver119@gmail.com
//

#include <graph/execution/LogicEnter.h>
#include <Status.h>


namespace nd4j {
    namespace graph {
        Nd4jStatus LogicEnter::processNode(Graph *graph, Node *node) {
            // this op replicates input variable into the frame. basically happens once for single loop.
            // sure, if there's inner loop within outer loop, it'll be called once for outer loop and multiple times for inner loop

            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();

            // basically, first non-null variable is our target
            for (int e = 0; e < node->input()->size(); e++) {
                auto inputAddr = node->input()->at(e);

                if (__variableSpace->hasVariable(inputAddr)) {
                    auto var = __variableSpace->getVariable(inputAddr);
                    if (var->hasNDArray()) {
                        Variable *lvar = nullptr;
                        if (__variableSpace->hasVariable(node->id(), 0))
                            lvar = __variableSpace->getVariable(node->id(), 0);
                        else
                            lvar = new Variable(nullptr, node->getName()->c_str(), node->id(), 0);

                        auto array = var->getNDArray();
                        lvar->setNDArray(array);
                        lvar->markReadOnly(true);

                        break;
                    } else if (var->hasNDArrayList()) {
                        Variable *lvar = nullptr;
                        if (__variableSpace->hasVariable(node->id(), 0))
                            lvar = __variableSpace->getVariable(node->id(), 0);
                        else
                            lvar = new Variable(nullptr, node->getName()->c_str(), node->id(), 0);

                        auto list = var->getNDArrayList();
                        lvar->setNDArrayList(list);
                        lvar->markReadOnly(true);

                        break;
                    } else {
                        // FIXME: can we really have third case here?
                        continue;
                    }
                }
            }

            return nd4j::Status::OK();
        }
    }
}