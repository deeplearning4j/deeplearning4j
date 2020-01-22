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

package org.deeplearning4j.arbiter.optimize.genetic.crossover;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.RandomTwoParentSelection;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class RandomTwoParentSelectionTests extends BaseDL4JTest {
    @Test
    public void RandomTwoParentSelection_ShouldReturnTwoDifferentParents() {
        RandomGenerator rng = new TestRandomGenerator(new int[] {1, 1, 1, 0}, null);
        RandomTwoParentSelection sut = new RandomTwoParentSelection(rng);

        List<Chromosome> population = new ArrayList<>();
        population.add(new Chromosome(new double[] {1, 1, 1}, 1.0));
        population.add(new Chromosome(new double[] {2, 2, 2}, 2.0));
        population.add(new Chromosome(new double[] {3, 3, 3}, 3.0));
        sut.initializeInstance(population);

        double[][] result = sut.selectParents();

        Assert.assertSame(population.get(1).getGenes(), result[0]);
        Assert.assertSame(population.get(0).getGenes(), result[1]);
    }
}
