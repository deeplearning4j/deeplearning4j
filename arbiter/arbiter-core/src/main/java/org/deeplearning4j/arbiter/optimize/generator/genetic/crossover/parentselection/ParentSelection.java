/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the terms of the Apache License, Version 2.0
 * which is available at https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;

import java.util.List;

/**
 * Abstract class for all parent selection behaviors
 *
 * @author Alexandre Boulanger
 */
public abstract class ParentSelection {
    protected List<Chromosome> population;

    /**
     * Will be called by the crossover operator once the population model is instantiated.
     */
    public void initializeInstance(List<Chromosome> population) {
        this.population = population;
    }

    /**
     * Performs the parent selection
     *
     * @return An array of parents genes. The outer array are the parents, and the inner array are the genes.
     */
    public abstract double[][] selectParents();
}
