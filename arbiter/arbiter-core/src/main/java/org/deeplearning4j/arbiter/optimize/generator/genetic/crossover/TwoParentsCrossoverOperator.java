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

package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

/**
 * Abstract class for all crossover operators that applies to two parents.
 *
 * @author Alexandre Boulanger
 */
public abstract class TwoParentsCrossoverOperator extends CrossoverOperator {

    protected final TwoParentSelection parentSelection;

    /**
     * @param parentSelection A parent selection that selects two parents.
     */
    protected TwoParentsCrossoverOperator(TwoParentSelection parentSelection) {
        this.parentSelection = parentSelection;
    }

    /**
     * Will be called by the selection operator once the population model is instantiated.
     */
    @Override
    public void initializeInstance(PopulationModel populationModel) {
        super.initializeInstance(populationModel);
        parentSelection.initializeInstance(populationModel.getPopulation());
    }
}
