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
// Created by raver119 on 15.10.2017.
//

#include "ops/declarable/LogicOp.h"

namespace nd4j {
    namespace ops {

        template <typename T>
        LogicOp<T>::LogicOp(const char *name) : DeclarableOp<T>::DeclarableOp(name, true) {
            // just using DeclarableOp constructor
            //this->_descriptor->
        }

        template <typename T>
        Nd4jStatus LogicOp<T>::validateAndExecute(nd4j::graph::Context<T> &block) {
            nd4j_logger("WARNING: LogicOps should NOT be ever called\n", "");
            return ND4J_STATUS_BAD_INPUT;
        }

        template <typename T>
        ShapeList* LogicOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            // FIXME: we probably want these ops to evaluate scopes
            return SHAPELIST();
        }

        template class ND4J_EXPORT LogicOp<float>;
        template class ND4J_EXPORT LogicOp<float16>;
        template class ND4J_EXPORT LogicOp<double>;
        template class ND4J_EXPORT LogicOp<int>;
        template class ND4J_EXPORT LogicOp<Nd4jLong>;
    }
}