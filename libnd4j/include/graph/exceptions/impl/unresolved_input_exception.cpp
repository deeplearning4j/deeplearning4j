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

#include <graph/exceptions/unresolved_input_exception.h>
#include <helpers/StringUtils.h>

namespace sd {
    namespace graph {
        unresolved_input_exception::unresolved_input_exception(std::string message) : std::runtime_error(message) {
            //
        }

        unresolved_input_exception unresolved_input_exception::build(std::string message, int nodeId, std::pair<int, int> &varIndex) {
            auto node = StringUtils::valueToString<int>(nodeId);
            auto varId = StringUtils::valueToString<int>(varIndex.first);
            auto outputIdx = StringUtils::valueToString<int>(varIndex.second);
            message += "; Node: [" + node +":0]; Variable: [" + varId + ":" +  outputIdx + "]";
            return unresolved_input_exception(message);
        }

        unresolved_input_exception unresolved_input_exception::build(std::string message, std::pair<int, int> &varIndex) {
            auto nodeId = StringUtils::valueToString<int>(varIndex.first);
            auto outputIdx = StringUtils::valueToString<int>(varIndex.second);
            message += "; Variable: [" + nodeId + ":" +  outputIdx + "]";
            return unresolved_input_exception(message);
        }

        unresolved_input_exception unresolved_input_exception::build(std::string message, std::string &varName) {
            message += "; Variable: [" + varName + "]";
            return unresolved_input_exception(message);
        }
    }
}