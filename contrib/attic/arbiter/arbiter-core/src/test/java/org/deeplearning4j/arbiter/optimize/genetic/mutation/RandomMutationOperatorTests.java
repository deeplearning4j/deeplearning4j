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

package org.deeplearning4j.arbiter.optimize.genetic.mutation;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.RandomMutationOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;

import java.lang.reflect.Field;

public class RandomMutationOperatorTests extends BaseDL4JTest {
    @Test
    public void RandomMutationOperator_DefaultBuild_ShouldNotBeNull() {
        RandomMutationOperator sut = new RandomMutationOperator.Builder().build();
        Assert.assertNotNull(sut);
    }

    @Test
    public void RandomMutationOperator_BuildWithMutationRate_ShouldUseSuppliedRate() throws Exception {
        RandomMutationOperator sut = new RandomMutationOperator.Builder().mutationRate(0.123).build();

        Field f = sut.getClass().getDeclaredField("mutationRate");
        f.setAccessible(true);
        Double mutationRate = (Double) f.get(sut);

        Assert.assertEquals(0.123, mutationRate, 0.0);
    }

    @Test
    public void RandomMutationOperator_BelowMutationRate_ShouldNotMutate() {
        double[] randomNumbers = new double[] {0.1, 1.0, 1.0};

        RandomMutationOperator sut = new RandomMutationOperator.Builder().mutationRate(0.1)
                        .randomGenerator(new TestRandomGenerator(null, randomNumbers)).build();

        double[] genes = new double[] {-1.0, -1.0, -1.0};
        boolean hasMutated = sut.mutate(genes);

        Assert.assertFalse(hasMutated);
        Assert.assertArrayEquals(new double[]{-1.0, -1.0, -1.0}, genes, 0.0);
    }

    @Test
    public void RandomMutationOperator_AboveMutationRate_ShouldMutate() {
        double[] randomNumbers = new double[] {0.099, 0.123, 1.0, 1.0};

        RandomMutationOperator sut = new RandomMutationOperator.Builder().mutationRate(0.1)
                        .randomGenerator(new TestRandomGenerator(null, randomNumbers)).build();

        double[] genes = new double[] {-1.0, -1.0, -1.0};
        boolean hasMutated = sut.mutate(genes);

        Assert.assertTrue(hasMutated);
        Assert.assertArrayEquals(new double[]{0.123, -1.0, -1.0}, genes, 0.0);
    }
}
