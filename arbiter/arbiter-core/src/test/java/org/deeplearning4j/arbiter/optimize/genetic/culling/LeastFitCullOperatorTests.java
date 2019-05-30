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

package org.deeplearning4j.arbiter.optimize.genetic.culling;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.LeastFitCullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class LeastFitCullOperatorTests {

    @Test
    public void LeastFitCullingOperation_ShouldCullLastElements() {
        LeastFitCullOperator sut = new LeastFitCullOperator(0.50);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                        .populationSize(10).build();
        sut.initializeInstance(populationModel);

        List<Chromosome> originalChromosomes = new ArrayList<>();
        for (int i = 0; i < 10; ++i) {
            originalChromosomes.add(new Chromosome(null, (double) i));
        }

        List<Chromosome> chromosomes = populationModel.getPopulation();
        for (int i = 0; i < 10; ++i) {
            chromosomes.add(originalChromosomes.get(i));
        }

        sut.cullPopulation();

        Assert.assertEquals(5, chromosomes.size());
        for (int i = 0; i < 5; ++i) {
            Assert.assertSame(originalChromosomes.get(i), chromosomes.get(i));
        }
    }


}
