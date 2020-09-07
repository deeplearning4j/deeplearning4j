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

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.RatioCullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;

public class RatioCullOperatorTests extends BaseDL4JTest {

    static class TestRatioCullOperator extends RatioCullOperator {

        public TestRatioCullOperator() {
            super();
        }

        public TestRatioCullOperator(double ratio) {
            super(ratio);
        }

        public List<Chromosome> getPopulation() {
            return population;
        }

        @Override
        public void cullPopulation() {
            throw new NotImplementedException("Not implemented");
        }

        public double getCullRatio() {
            return cullRatio;
        }
    }

    @Test
    public void RatioCullingOperation_ctorWithCullRatio_ShouldHaveParamRatio() {
        TestRatioCullOperator sut = new TestRatioCullOperator(0.123);

        Assert.assertEquals(0.123, sut.getCullRatio(), 0.0);
    }

    @Test
    public void RatioCullingOperation_initialize_shouldSetCulledSizeAndPopulation() throws IllegalAccessException {
        TestRatioCullOperator sut = new TestRatioCullOperator(0.50);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                        .populationSize(10).build();
        sut.initializeInstance(populationModel);

        Assert.assertSame(populationModel.getPopulation(), sut.getPopulation());
        Assert.assertEquals(5, sut.getCulledSize());
    }

}
