/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.arbiter.optimize.genetic;

import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

public class TestCrossoverOperator extends CrossoverOperator {

    private final CrossoverResult[] results;
    private int resultIdx = 0;

    public PopulationModel getPopulationModel() {
        return populationModel;
    }

    public TestCrossoverOperator(CrossoverResult[] results) {
        this.results = results;
    }

    @Override
    public CrossoverResult crossover() {
        return results[resultIdx++];
    }
}
