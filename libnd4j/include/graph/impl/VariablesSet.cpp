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
// Created by raver119 on 15/11/17.
//

#include <graph/VariablesSet.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        Nd4jStatus VariablesSet<T>::status() {
            return _status;
        }

        template <typename T>
        int VariablesSet<T>::size() {
            return _holder.size();
        }

        template <typename T>
        void VariablesSet<T>::push_back(Variable<T> *variable) {
            _holder.push_back(variable);
        }

        template <typename T>
        Variable<T> *VariablesSet<T>::at(int index) {
            return _holder.at(index);
        }

        template<typename T>
        VariablesSet<T>::VariablesSet(Nd4jStatus status) {
            _status = status;
        }

        template<typename T>
        VariablesSet<T>::~VariablesSet() {
            for (auto v: _holder)
                delete v;
        }

        template class ND4J_EXPORT VariablesSet<float>;
        template class ND4J_EXPORT VariablesSet<float16>;
        template class ND4J_EXPORT VariablesSet<double>;
        template class ND4J_EXPORT VariablesSet<int>;
        template class ND4J_EXPORT VariablesSet<Nd4jLong>;
    }
}
