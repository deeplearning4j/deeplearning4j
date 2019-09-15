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
// @author raver119@gmail.com
//

#ifndef SAMEDIFF_SDVARIABLE_H
#define SAMEDIFF_SDVARIABLE_H

#include <NDArray.h>
#include <graph/Node.h>
#include <graph/Variable.h>

namespace samediff {
    class SameDiff;

    class Variable {
    protected:
        // TODO: use shared_ptr here
        nd4j::graph::Node* _node = nullptr;
        SameDiff* _sd;

        // output index within node
        int _index = 0;
    public:
        Variable() = default;
        ~Variable() = default;

        Variable(SameDiff &sd, nd4j::graph::Node *node, int index = 0);

        SameDiff* sd() const;
        int nodeId() const;

        // basic arithmetic operators
        Variable operator+(const Variable& other) const;

        //
        nd4j::NDArray array();
    };
}

ND4J_EXPORT samediff::Variable operator+(const float&, const samediff::Variable&);
ND4J_EXPORT samediff::Variable operator+(const samediff::Variable&, const float&);

#endif //SAMEDIFF_SDVARIABLE_H
