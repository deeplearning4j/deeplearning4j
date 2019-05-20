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

package org.deeplearning4j.arbiter.optimize.generator.genetic.culling;

import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

/**
 * The cull operator will remove from the population the least desirables chromosomes.
 *
 * @author Alexandre Boulanger
 */
public interface CullOperator {
    /**
     * Will be called by the population model once created.
     */
    void initializeInstance(PopulationModel populationModel);

    /**
     * Cull the population to the culled size.
     */
    void cullPopulation();

    /**
     * @return The target population size after culling.
     */
    int getCulledSize();
}
