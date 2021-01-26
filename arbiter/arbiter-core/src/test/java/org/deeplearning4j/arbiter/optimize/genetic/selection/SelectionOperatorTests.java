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

package org.deeplearning4j.arbiter.optimize.genetic.selection;

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.SelectionOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;

public class SelectionOperatorTests extends BaseDL4JTest {
    private class TestSelectionOperator extends SelectionOperator {

        public PopulationModel getPopulationModel() {
            return populationModel;
        }

        public ChromosomeFactory getChromosomeFactory() {
            return chromosomeFactory;
        }

        @Override
        public double[] buildNextGenes() {
            throw new NotImplementedException("Not implemented");
        }
    }

    @Test
    public void SelectionOperator_InitializeInstance_ShouldInitializeFields() {
        TestSelectionOperator sut = new TestSelectionOperator();

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel =
                        new PopulationModel.Builder().populationInitializer(populationInitializer).build();
        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        sut.initializeInstance(populationModel, chromosomeFactory);

        Assert.assertSame(populationModel, sut.getPopulationModel());
        Assert.assertSame(chromosomeFactory, sut.getChromosomeFactory());
    }
}
