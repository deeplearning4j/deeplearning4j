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

package org.deeplearning4j.arbiter.optimize.generator.genetic;

/**
 * A factory that builds new chromosomes. Used by the GeneticSearchCandidateGenerator.
 *
 * @author Alexandre Boulanger
 */
public class ChromosomeFactory {
    private int chromosomeLength;

    /**
     * Called by the GeneticSearchCandidateGenerator.
     */
    public void initializeInstance(int chromosomeLength) {
        this.chromosomeLength = chromosomeLength;
    }

    /**
     * Create a new instance of a Chromosome
     *
     * @param genes The genes
     * @param fitness The fitness score
     * @return A new instance of Chromosome
     */
    public Chromosome createChromosome(double[] genes, double fitness) {
        return new Chromosome(genes, fitness);
    }

    /**
     * @return The number of genes in a chromosome
     */
    public int getChromosomeLength() {
        return chromosomeLength;
    }
}
