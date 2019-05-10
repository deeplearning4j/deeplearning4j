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

#ifndef DEV_TESTS_BOOLPARAMETERS_H
#define DEV_TESTS_BOOLPARAMETERS_H

#include <map>
#include <vector>
#include <string>
#include "Parameters.h"
#include "ParametersSpace.h"

namespace nd4j {
    class BoolParameters : public ParametersSpace {
    protected:

    public:
        BoolParameters(std::string name) : ParametersSpace() {
            _name = name;
        }

        std::vector<int> evaluate() override {
            std::vector<int> result;
            result.emplace_back(0);
            result.emplace_back(1);
            return result;
        }
    };
}

#endif //DEV_TESTS_PARAMETERSPACE_H