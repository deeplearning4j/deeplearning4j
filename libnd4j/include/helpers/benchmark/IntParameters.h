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

#ifndef DEV_TESTS_INTPARAMETERS_H
#define DEV_TESTS_INTPARAMETERS_H

#include <map>
#include <vector>
#include <string>
#include "Parameters.h"
#include "ParametersSpace.h"

namespace nd4j {
    class IntParameters : public ParametersSpace {
    protected:
        int _start;
        int _stop;
        int _step;

    public:
        IntParameters(std::string name, int start, int stop, int step = 1) : ParametersSpace() {
            _start = start;
            _stop = stop;
            _step = step;
            _name = name;
        }

        std::vector<int> evaluate() override {
            std::vector<int> result;
            for (int e = _start; e <= _stop; e += _step) {
               result.emplace_back(e);
            }
            return result;
        }
    };
}

#endif //DEV_TESTS_INTPARAMETERS_H