/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#ifndef DEV_TESTS_PARAMETERS_H
#define DEV_TESTS_PARAMETERS_H

#include <map>
#include <string>
#include <vector>

namespace nd4j {
    class Parameters {
    private:
        std::map<std::string, int> _intParams;
        std::map<std::string, bool> _boolParams;
        std::map<std::string, std::vector<int>> _arrayParams;
    public:
        Parameters() = default;

        Parameters* addIntParam(std::string string, int param);
        Parameters* addIntParam(std::initializer_list<std::string> strings, std::initializer_list<int> params);

        Parameters* addBoolParam(std::string string, bool param);
        Parameters* addBoolParam(std::initializer_list<std::string> strings, std::initializer_list<bool> params);

        Parameters* addArrayParam(std::string string, std::initializer_list<int> param);
        Parameters* addArrayParam(std::initializer_list<std::string> strings, std::initializer_list<std::initializer_list<int>> params);

        int getIntParam(std::string string) const ;
        bool getBoolParam(std::string string) const;
        std::vector<int> getArrayParam(std::string string) const;
    };
}

#endif //DEV_TESTS_PARAMETERS_H