/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <graph/exceptions/unresolved_output_exception.h>
#include <helpers/StringUtils.h>
#include <utility>

namespace sd {
    namespace graph {
        unresolved_output_exception::unresolved_output_exception(std::string message) : std::runtime_error(message) {
            //
        }

        unresolved_output_exception unresolved_output_exception::build(std::string message, std::pair<int, int> &varIndex) {
            auto nodeId = StringUtils::valueToString<int>(varIndex.first);
            auto outputIdx = StringUtils::valueToString<int>(varIndex.second);
            message += "; Variable: [" + nodeId + ":" +  outputIdx + "]";
            return unresolved_output_exception(message);
        }

        unresolved_output_exception unresolved_output_exception::build(std::string message, int nodeId, int outputIndex) {
            std::pair<int, int> p(nodeId, outputIndex);
            return build(message, p);
        }

        unresolved_output_exception unresolved_output_exception::build(std::string message, std::string &varName, int outputIndex) {
            auto outputIdx = StringUtils::valueToString<int>(outputIndex);
            message += "; Variable: [" + varName + ":" + outputIdx + "]";
            return unresolved_output_exception(message);
        }
    }
}