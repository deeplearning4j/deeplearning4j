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

#ifndef SAMEDIFF_TUPLE_H
#define SAMEDIFF_TUPLE_H

#include <graph/Node.h>
#include <samediff/Variable.h>

namespace samediff {
    class SameDiff;

    class Tuple {
    private:
        nd4j::graph::Node *_node;
        SameDiff *_sd;

    public:
        Tuple() = default;
        ~Tuple() = default;

        uint32_t size() const;

        Variable at(uint32_t index) const;

        Variable operator[](const uint32_t index) const;
    };
}


#endif //SAMEDIFF_TUPLE_H
