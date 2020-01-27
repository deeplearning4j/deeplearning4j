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

package org.deeplearning4j.arbiter.optimize.genetic.selection;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.CullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.exceptions.GeneticGenerationException;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.MutationOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.GeneticSelectionOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestCrossoverOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestMutationOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import static org.junit.Assert.assertArrayEquals;

public class GeneticSelectionOperatorTests extends BaseDL4JTest {

    private class TestCullOperator implements CullOperator {

        private final int culledSize;

        public TestCullOperator(int culledSize) {

            this.culledSize = culledSize;
        }

        @Override
        public void initializeInstance(PopulationModel populationModel) {

        }

        @Override
        public void cullPopulation() {
            throw new NotImplementedException();
        }

        @Override
        public int getCulledSize() {
            return culledSize;
        }
    }

    private class GeneticSelectionOperatorTestsMutationOperator implements MutationOperator {

        private boolean mutateResult;

        public GeneticSelectionOperatorTestsMutationOperator(boolean mutateResult) {

            this.mutateResult = mutateResult;
        }

        @Override
        public boolean mutate(double[] genes) {
            return mutateResult;
        }
    }

    private class GeneticSelectionOperatorTestsCrossoverOperator extends CrossoverOperator {

        private CrossoverResult result;

        public GeneticSelectionOperatorTestsCrossoverOperator(CrossoverResult result) {

            this.result = result;
        }

        @Override
        public CrossoverResult crossover() {
            return result;
        }
    }

    @Test
    public void GeneticSelectionOperator_PopulationNotReadyToBreed_ShouldReturnRandomGenes() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] {123.0});

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperator cullOperator = new TestCullOperator(1000);
        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                        .cullOperator(cullOperator).build();
        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        chromosomeFactory.initializeInstance(1);
        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder().randomGenerator(rng).build();
        sut.initializeInstance(populationModel, chromosomeFactory);

        double[] newGenes = sut.buildNextGenes();

        Assert.assertEquals(1, newGenes.length);
        Assert.assertEquals(123.0, newGenes[0], 0.0);
    }

    @Test
    public void GeneticSelectionOperator_NoModificationOnFirstTry() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] {123.0});

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperator cullOperator = new TestCullOperator(-1);

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                        .cullOperator(cullOperator).build();

        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        chromosomeFactory.initializeInstance(1);

        CrossoverResult[] crossoverResults = new CrossoverResult[2];
        crossoverResults[0] = new CrossoverResult(false, new double[0]);
        crossoverResults[1] = new CrossoverResult(true, new double[0]);
        TestCrossoverOperator crossoverOperator = new TestCrossoverOperator(crossoverResults);

        boolean[] mutationResults = new boolean[] {false, false};
        TestMutationOperator mutationOperator = new TestMutationOperator(mutationResults);

        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder().randomGenerator(rng)
                        .crossoverOperator(crossoverOperator).mutationOperator(mutationOperator).build();
        sut.initializeInstance(populationModel, chromosomeFactory);

        double[] newGenes = sut.buildNextGenes();

        Assert.assertSame(crossoverResults[1].getGenes(), newGenes);
    }

    @Test
    public void GeneticSelectionOperator_MutationNoModificationOnFirstTry() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] {123.0});

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperator cullOperator = new TestCullOperator(-1);

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                        .cullOperator(cullOperator).build();

        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        chromosomeFactory.initializeInstance(1);

        CrossoverResult[] crossoverResults = new CrossoverResult[3];
        crossoverResults[0] = new CrossoverResult(false, new double[0]);
        crossoverResults[1] = new CrossoverResult(false, new double[0]);
        crossoverResults[2] = new CrossoverResult(true, new double[0]);
        TestCrossoverOperator crossoverOperator = new TestCrossoverOperator(crossoverResults);

        boolean[] mutationResults = new boolean[] {false, false, true};
        TestMutationOperator mutationOperator = new TestMutationOperator(mutationResults);

        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder().randomGenerator(rng)
                        .crossoverOperator(crossoverOperator).mutationOperator(mutationOperator).build();
        sut.initializeInstance(populationModel, chromosomeFactory);

        double[] newGenes = sut.buildNextGenes();

        Assert.assertSame(crossoverResults[2].getGenes(), newGenes);
    }

    @Test
    public void GeneticSelectionOperator_ShouldNotBuildDuplicates() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] {123.0});

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperator cullOperator = new TestCullOperator(-1);

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                        .cullOperator(cullOperator).build();

        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        chromosomeFactory.initializeInstance(1);

        CrossoverResult[] crossoverResults = new CrossoverResult[3];
        crossoverResults[0] = new CrossoverResult(true, new double[] {1.0});
        crossoverResults[1] = new CrossoverResult(true, new double[] {1.0});
        crossoverResults[2] = new CrossoverResult(true, new double[] {2.0});
        TestCrossoverOperator crossoverOperator = new TestCrossoverOperator(crossoverResults);

        boolean[] mutationResults = new boolean[] {false, false, false};
        TestMutationOperator mutationOperator = new TestMutationOperator(mutationResults);

        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder().randomGenerator(rng)
                        .crossoverOperator(crossoverOperator).mutationOperator(mutationOperator).build();
        sut.initializeInstance(populationModel, chromosomeFactory);

        double[] newGenes = sut.buildNextGenes();
        assertArrayEquals(crossoverResults[0].getGenes(), newGenes, 1e-6);

        newGenes = sut.buildNextGenes();
        assertArrayEquals(crossoverResults[2].getGenes(), newGenes, 1e-6);
    }

    @Test(expected = GeneticGenerationException.class)
    public void GeneticSelectionOperator_CrossoverAndMutationCantGenerateNew_ShouldThrow() {
        TestCullOperator cullOperator = new TestCullOperator(-1);

        PopulationModel populationModel = new PopulationModel.Builder().cullOperator(cullOperator).build();

        MutationOperator mutationOperator = new GeneticSelectionOperatorTestsMutationOperator(false);
        CrossoverOperator crossoverOperator =
                        new GeneticSelectionOperatorTestsCrossoverOperator(new CrossoverResult(false, null));

        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder().crossoverOperator(crossoverOperator)
                        .mutationOperator(mutationOperator).build();
        sut.initializeInstance(populationModel, null);

        sut.buildNextGenes();
    }

    @Test(expected = GeneticGenerationException.class)
    public void GeneticSelectionOperator_CrossoverAndMutationAlwaysGenerateSame_ShouldThrow() {
        TestCullOperator cullOperator = new TestCullOperator(-1);

        PopulationModel populationModel = new PopulationModel.Builder().cullOperator(cullOperator).build();

        MutationOperator mutationOperator = new GeneticSelectionOperatorTestsMutationOperator(false);
        CrossoverOperator crossoverOperator = new GeneticSelectionOperatorTestsCrossoverOperator(
                        new CrossoverResult(true, new double[] {1.0}));

        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder().crossoverOperator(crossoverOperator)
                        .mutationOperator(mutationOperator).build();
        sut.initializeInstance(populationModel, null);

        // This call is used to add the genes to the previousGenes collection
        sut.buildNextGenes();

        sut.buildNextGenes();
    }
}
