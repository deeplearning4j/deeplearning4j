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

#include <graph/execution/LogicLoopCond.h>


namespace nd4j {
    namespace graph {
        Nd4jStatus LogicLoopCond::processNode(Graph *graph, Node *node) {
            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();

            Context ctx(node->getContextPrototype(), __variableSpace);
            auto input = ctx.variable(0)->getNDArray();

            std::pair<int, int> pair0(node->id(), 0);

            if (!__variableSpace->hasVariable(pair0))
                __variableSpace->putVariable(pair0, new Variable(nullptr, nullptr, node->id(), 0));

            __variableSpace->getVariable(pair0)->setNDArray(input);
            __variableSpace->getVariable(pair0)->markRemovable(false);

            // pass further
            if (input->getScalar(0) > 0.0) {
                // if condition is TRUE body will be invoked some time soon
     //           __flowPath->markFrameActive(node->getFrameId(), true);
                //__flowPath->i
            } else {
                // body won't be activated
     //           __flowPath->markFrameActive(node->getFrameId(), false);
            }

            return ND4J_STATUS_OK;
        }
    }
}