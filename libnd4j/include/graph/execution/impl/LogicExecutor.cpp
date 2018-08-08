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


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicExecutor<T>::processNode(Graph<T> *graph, Node<T> *node) {
            switch (node->opNum()) {
                case 0:
                    return LogicWhile<T>::processNode(graph, node);
                case 10:
                    return LogicScope<T>::processNode(graph, node);
                case 20:
                    return LogicConditional<T>::processNode(graph, node);
                case 30:
                    return LogicSwitch<T>::processNode(graph, node);
                case 40:
                    return LogicReturn<T>::processNode(graph, node);
                case 50:
                    return LogicExpose<T>::processNode(graph, node);
                case 60:
                    return LogicMerge<T>::processNode(graph, node);
                case 70:
                    return LogicLoopCond<T>::processNode(graph, node);
                case 80:
                    return LogicNextIeration<T>::processNode(graph, node);
                case 90:
                    return LogicExit<T>::processNode(graph, node);
                case 100:
                    return LogicEnter<T>::processNode(graph, node);
            }

            if (node->getName() == nullptr) {
                nd4j_printf("Unknown LogicOp used at node [%i]: [%i]\n", node->id(), node->opNum());
            } else {
                nd4j_printf("Unknown LogicOp used at node [%i:<%s>]: [%i]\n", node->id(), node->getName()->c_str(), node->opNum());
            }
            return ND4J_STATUS_BAD_INPUT;
        }

        template class ND4J_EXPORT LogicExecutor<float>;
        template class ND4J_EXPORT LogicExecutor<float16>;
        template class ND4J_EXPORT LogicExecutor<double>;
        template class ND4J_EXPORT LogicExecutor<int>;
        template class ND4J_EXPORT LogicExecutor<Nd4jLong>;
    }
}