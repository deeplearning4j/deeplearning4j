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
// Created by raver119 on 28.10.2017.
//

#include "graph/execution/LogicReturn.h"
#include <helpers/EnumUtils.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        Nd4jStatus LogicReturn<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();

            for (int e = 0; e < node->input()->size(); e++) {
                auto inputAddr = node->input()->at(e);
                auto outputAddr = node->output()->at(e);

                // FIXME!!
                outputAddr.second = e;

                if (Environment::getInstance()->isDebugAndVerbose())
                    nd4j_debug("Return input: <%i, %i>; Return output: <%i, %i>\n", inputAddr.first, inputAddr.second, outputAddr.first, outputAddr.second);

                auto varIn = __variableSpace->getVariable(inputAddr);
                auto varOut = __variableSpace->getVariable(outputAddr);

                nd4j_debug("Returning varType: [%s]\n", EnumUtils::_VariableTypeToString(varIn->variableType()));

                // FIXME: this is obviously wrong, we should keep depth track for backprop here
                varOut->getNDArray()->assign(varIn->getNDArray());

                if (Environment::getInstance()->isDebugAndVerbose())
                    nd4j_debug("In after: [%f]; Out after: [%f]\n", varIn->getNDArray()->meanNumber(), varOut->getNDArray()->meanNumber());
            }

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicReturn<float>;
        template class ND4J_EXPORT LogicReturn<float16>;
        template class ND4J_EXPORT LogicReturn<double>;
        template class ND4J_EXPORT LogicReturn<int>;
        template class ND4J_EXPORT LogicReturn<Nd4jLong>;
    }
}
