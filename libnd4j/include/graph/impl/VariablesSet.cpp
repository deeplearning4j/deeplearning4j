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
        Nd4jStatus VariablesSet::status() {
            return _status;
        }

        int VariablesSet::size() {
            return _holder.size();
        }

        void VariablesSet::push_back(Variable *variable) {
            _holder.push_back(variable);
        }

        Variable *VariablesSet::at(int index) {
            return _holder.at(index);
        }

        VariablesSet::VariablesSet(Nd4jStatus status) {
            _status = status;
        }

        VariablesSet::~VariablesSet() {
            for (auto v: _holder)
                delete v;
        }
    }
}
