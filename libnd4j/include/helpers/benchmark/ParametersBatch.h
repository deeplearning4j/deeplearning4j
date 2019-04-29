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

#ifndef DEV_TESTS_PARAMETERSBATCH_H
#define DEV_TESTS_PARAMETERSBATCH_H

#include "ParametersSpace.h"
#include <vector>
#include <shape.h>

namespace nd4j {
    class ParametersBatch {
    protected:
        std::vector<ParametersSpace*> _spaces;
    public:
        ParametersBatch() = default;
        ParametersBatch(std::initializer_list<ParametersSpace*> spaces) {
            _spaces = spaces;
        }

        ParametersBatch(std::vector<ParametersSpace*> spaces) {
            _spaces = spaces;
        }


        std::vector<Parameters> parameters() {
            std::vector<Parameters> result;
            std::vector<std::vector<int>> vectors;
            int totalIterations = 1;

            // hehe
            Nd4jLong xCoords[MAX_RANK];
            Nd4jLong xShape[MAX_RANK];
            int xRank = _spaces.size();

            for (int e = 0; e < _spaces.size(); e++) {
                auto space = _spaces[e];
                auto values = space->evaluate();
                vectors.emplace_back(values);

                totalIterations *= values.size();
                xShape[e] = values.size();
            }

            //nd4j_printf("Total Iterations: %i\n", totalIterations);



            for (int i = 0; i < totalIterations; i++) {
                shape::ind2subC(xRank, xShape, i, totalIterations, xCoords);

                Parameters params;
                for (int j = 0; j < xRank; j++) {
                    int value = vectors[j][xCoords[j]];
                    std::string name = _spaces[j]->name();
                    params.addIntParam(name, value);
                }

                result.emplace_back(params);
            }


            return result;
        }
    };
}

#endif //DEV_TESTS_PARAMETERSBATCH_H