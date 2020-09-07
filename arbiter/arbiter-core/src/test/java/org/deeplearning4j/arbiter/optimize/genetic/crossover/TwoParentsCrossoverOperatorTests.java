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

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.TwoParentsCrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.genetic.TestParentSelection;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;

public class TwoParentsCrossoverOperatorTests extends BaseDL4JTest {

    class TestTwoParentsCrossoverOperator extends TwoParentsCrossoverOperator {

        public TestTwoParentsCrossoverOperator(TwoParentSelection parentSelection) {
            super(parentSelection);
        }

        public TwoParentSelection getParentSelection() {
            return parentSelection;
        }

        @Override
        public CrossoverResult crossover() {
            throw new NotImplementedException("Not implemented");
        }
    }

    @Test
    public void TwoParentsCrossoverOperator_ctor_ShouldInitParentSelection() {
        TestParentSelection parentSelection = new TestParentSelection();
        TestTwoParentsCrossoverOperator sut = new TestTwoParentsCrossoverOperator(parentSelection);

        Assert.assertSame(parentSelection, sut.getParentSelection());
    }

    @Test
    public void TwoParentsCrossoverOperator_initializeInstanceShouldInitializeParentSelection() {
        TestParentSelection parentSelection = new TestParentSelection();
        TestTwoParentsCrossoverOperator sut = new TestTwoParentsCrossoverOperator(parentSelection);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();
        PopulationModel populationModel =
                        new PopulationModel.Builder().populationInitializer(populationInitializer).build();

        sut.initializeInstance(populationModel);

        Assert.assertTrue(parentSelection.hasBeenInitialized);
    }

}
