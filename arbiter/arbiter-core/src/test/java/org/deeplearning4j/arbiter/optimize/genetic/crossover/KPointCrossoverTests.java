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
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.KPointCrossover;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;
import org.deeplearning4j.arbiter.optimize.genetic.TestParentSelection;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;

public class KPointCrossoverTests extends BaseDL4JTest {

    @Test
    public void KPointCrossover_BelowCrossoverRate_ShouldReturnParent0() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] {1.0});

        double[][] parents = new double[2][];
        parents[0] = new double[] {0.0};
        parents[1] = new double[] {1.0};
        TwoParentSelection parentSelection = new TestParentSelection(parents);
        KPointCrossover sut = new KPointCrossover.Builder().randomGenerator(rng).crossoverRate(0.0)
                        .parentSelection(parentSelection).build();

        CrossoverResult result = sut.crossover();

        Assert.assertFalse(result.isModified());
        Assert.assertSame(parents[0], result.getGenes());
    }

    @Test
    public void KPointCrossover_FixedNumberOfCrossovers() {
        RandomGenerator rng = new TestRandomGenerator(new int[] {0, 1}, new double[] {0.0});

        double[][] parents = new double[3][];
        parents[0] = new double[] {0.0, 0.0, 0.0, 0.0, 0.0};
        parents[1] = new double[] {1.0, 1.0, 1.0, 1.0, 1.0};
        parents[2] = new double[] {2.0, 2.0, 2.0, 2.0, 2.0};
        TwoParentSelection parentSelection = new TestParentSelection(parents);
        KPointCrossover sut = new KPointCrossover.Builder().randomGenerator(rng).crossoverRate(1.0)
                        .parentSelection(parentSelection).numCrossovers(2).build();

        CrossoverResult result = sut.crossover();

        Assert.assertTrue(result.isModified());
        for (double x : result.getGenes()) {
            Assert.assertTrue(x == 0.0 || x == 1.0);
        }
    }
}
