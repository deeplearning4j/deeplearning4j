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

package org.deeplearning4j.arbiter.optimize.genetic;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;

import java.util.List;

public class TestParentSelection extends TwoParentSelection {

    public boolean hasBeenInitialized;

    private final double[][] parents;

    public TestParentSelection(double[][] parents) {
        this.parents = parents;
    }

    public TestParentSelection() {
        this(null);
    }

    @Override
    public void initializeInstance(List<Chromosome> population) {
        super.initializeInstance(population);
        hasBeenInitialized = true;
    }

    @Override
    public double[][] selectParents() {
        return parents;
    }

    public List<Chromosome> getPopulation() {
        return population;
    }
}
