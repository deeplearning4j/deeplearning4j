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

package org.deeplearning4j.arbiter.optimize.genetic.crossover;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.ArithmeticCrossover;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.genetic.TestParentSelection;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;

public class ArithmeticCrossoverTests extends BaseDL4JTest {

    @Test
    public void ArithmeticCrossover_Crossover_OutsideCrossoverRate_ShouldReturnParent0() {
        double[][] parents = new double[2][];
        parents[0] = new double[] {1.0};
        parents[1] = new double[] {2.0};

        TestParentSelection parentSelection = new TestParentSelection(parents);

        RandomGenerator rng = new TestRandomGenerator(null, new double[] {1.0});

        ArithmeticCrossover sut =
                        new ArithmeticCrossover.Builder().parentSelection(parentSelection).randomGenerator(rng).build();
        CrossoverResult result = sut.crossover();

        Assert.assertFalse(result.isModified());
        Assert.assertEquals(1, result.getGenes().length);
        Assert.assertEquals(1.0, result.getGenes()[0], 0.001);
    }

    @Test
    public void ArithmeticCrossover_Crossover_WithinCrossoverRate_ShouldReturnLinearCombination() {
        double[][] parents = new double[2][];
        parents[0] = new double[] {1.0};
        parents[1] = new double[] {2.0};

        TestParentSelection parentSelection = new TestParentSelection(parents);

        RandomGenerator rng = new TestRandomGenerator(null, new double[] {0.1, 0.1});

        ArithmeticCrossover sut =
                        new ArithmeticCrossover.Builder().parentSelection(parentSelection).randomGenerator(rng).build();
        CrossoverResult result = sut.crossover();

        Assert.assertTrue(result.isModified());
        Assert.assertEquals(1, result.getGenes().length);
        Assert.assertEquals(1.9, result.getGenes()[0], 0.001);
    }

}
