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
// Created by raver on 8/31/2018.
//

#ifndef DEV_TESTS_GRAPH_EXECUTION_EXCEPTION_H
#define DEV_TESTS_GRAPH_EXECUTION_EXCEPTION_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <stdexcept>

namespace nd4j {
    namespace graph {
        class graph_execution_exception: public std::runtime_error {
        private:
            Nd4jLong _graphId;
        public:
            explicit graph_execution_exception(Nd4jLong graphId);

            Nd4jLong graphId();
        };
    }
}

#endif //DEV_TESTS_UNKNOWN_GRAPH_EXCEPTION_H
