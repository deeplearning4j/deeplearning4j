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
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.utils.CrossoverPointsGenerator;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;

import java.util.Deque;

public class CrossoverPointsGeneratorTests extends BaseDL4JTest {

    @Test
    public void CrossoverPointsGenerator_FixedNumberCrossovers() {
        RandomGenerator rng = new TestRandomGenerator(new int[] {0}, null);
        CrossoverPointsGenerator sut = new CrossoverPointsGenerator(10, 2, 2, rng);

        Deque<Integer> result = sut.getCrossoverPoints();

        Assert.assertEquals(3, result.size());
        int a = result.pop();
        int b = result.pop();
        int c = result.pop();
        Assert.assertTrue(a < b);
        Assert.assertTrue(b < c);
        Assert.assertEquals(Integer.MAX_VALUE, c);
    }
}
