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

#include <graph/execution/LogicNextIteration.h>


namespace nd4j {
    namespace graph {
        Nd4jStatus LogicNextIeration::processNode(Graph *graph, Node *node) {
            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();

            auto inputAddr = node->input()->at(0);

            auto var = __variableSpace->getVariable(inputAddr);

            Variable *lvar = nullptr;
            if (__variableSpace->hasVariable(node->id(), 0))
                lvar = __variableSpace->getVariable(node->id(), 0);
            else
                lvar = new Variable(nullptr, node->getName()->c_str(), node->id(), 0);

//            if (lvar->hasNDArray())
//                delete lvar->getNDArray();

            auto array = var->getNDArray();
            lvar->setNDArray(array);
            lvar->markReadOnly(true);

            return ND4J_STATUS_OK;
        }
    }
}