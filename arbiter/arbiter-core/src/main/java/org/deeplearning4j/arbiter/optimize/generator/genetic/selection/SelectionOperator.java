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

package org.deeplearning4j.arbiter.optimize.generator.genetic.selection;

import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

/**
 * An abstract class for all selection operators. Used by the GeneticSearchCandidateGenerator to generate new candidates.
 *
 * @author Alexandre Boulanger
 */
public abstract class SelectionOperator {
    protected PopulationModel populationModel;
    protected ChromosomeFactory chromosomeFactory;

    /**
     * Called by GeneticSearchCandidateGenerator
     */
    public void initializeInstance(PopulationModel populationModel, ChromosomeFactory chromosomeFactory) {

        this.populationModel = populationModel;
        this.chromosomeFactory = chromosomeFactory;
    }

    /**
     * Generate a new set of genes.
     */
    public abstract double[] buildNextGenes();
}
