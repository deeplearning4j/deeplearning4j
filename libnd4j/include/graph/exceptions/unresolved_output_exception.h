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

#ifndef DEV_TESTS_UNRESOLVED_OUTPUT_H
#define DEV_TESTS_UNRESOLVED_OUTPUT_H

#include <utility>
#include <string>
#include <stdexcept>

namespace sd {
    namespace graph {
        class unresolved_output_exception : public std::runtime_error {
        public:
            unresolved_output_exception(std::string message);
            ~unresolved_output_exception() = default;

            static unresolved_output_exception build(std::string message, int nodeId, int outputIndex);
            static unresolved_output_exception build(std::string message, std::pair<int, int> &varIndex);
            static unresolved_output_exception build(std::string message, std::string &varName, int outputIndex);
        };
    }
}


#endif //DEV_TESTS_UNRESOLVED_INPUT_H
