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

#ifndef DEV_TESTS_INTPOWERPARAMETERS_H
#define DEV_TESTS_INTPOWERPARAMETERS_H

#include <map>
#include <vector>
#include <string>
#include "Parameters.h"
#include "ParametersSpace.h"

namespace nd4j {
    class IntPowerParameters : public ParametersSpace {
    protected:
        int _base;
        int _start;
        int _stop;
        int _step;

    public:
        IntPowerParameters(std::string name, int base, int start, int stop, int step = 1) : ParametersSpace() {
            _base = base;
            _start = start;
            _stop = stop;
            _step = step;
            _name = name;
        }

        std::vector<int> evaluate() override {
            std::vector<int> result;
            for (int e = _start; e <= _stop; e += _step) {
               result.emplace_back(nd4j::math::nd4j_pow<double, double, int>(_base, e));
            }
            return result;
        }
    };
}

#endif //DEV_TESTS_INTPOWERPARAMETERS_H