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

#ifndef DEV_TESTS_PREDEFINEDPARAMETERS_H
#define DEV_TESTS_PREDEFINEDPARAMETERS_H

#include "ParametersSpace.h"

namespace nd4j {
    class PredefinedParameters : public ParametersSpace{
        std::vector<int> _params;
    public:
        PredefinedParameters(std::string name, std::initializer_list<int> parameters) : ParametersSpace() {
            _name = name;
            _params = parameters;
        }

        PredefinedParameters(std::string name, std::vector<int> parameters) : ParametersSpace() {
            _name = name;
            _params = parameters;
        }

        std::vector<int> evaluate() override {
            return _params;
        }
    };
}

#endif //DEV_TESTS_PREDEFINEDPARAMETERS_H