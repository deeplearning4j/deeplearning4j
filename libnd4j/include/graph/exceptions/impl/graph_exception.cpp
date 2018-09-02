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
// Created by raver on 9/1/2018.
//

#include <graph/exceptions/graph_exception.h>
#include <helpers/StringUtils.h>

namespace nd4j {
    namespace graph {
        graph_exception::graph_exception(std::string message, Nd4jLong graphId) : std::runtime_error(message) {
            this->_message = message;
            this->_graphId = graphId;
        }

        graph_exception::graph_exception(std::string message, std::string description, Nd4jLong graphId) : std::runtime_error(message) {
            this->_message = message;
            this->_description = description;
            this->_graphId = graphId;
        }

        graph_exception::graph_exception(std::string message, const char *description, Nd4jLong graphId) : std::runtime_error(message) {
            this->_message = message;
            this->_description = description;
            this->_graphId = graphId;
        }


        Nd4jLong graph_exception::graphId() {
            return _graphId;
        }

        const char* graph_exception::message() {
            return _message.c_str();
        }

        const char* graph_exception::description() {
            return _description.c_str();
        }
    }
}
