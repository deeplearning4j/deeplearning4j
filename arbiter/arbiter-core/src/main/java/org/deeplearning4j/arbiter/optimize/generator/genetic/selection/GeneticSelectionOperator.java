package org.deeplearning4j.arbiter.optimize.generator.genetic.selection;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.SinglePointCrossover;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.MutationOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.RandomMutationOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

import java.util.Arrays;

public class GeneticSelectionOperator extends SelectionOperator {

    private final static int PREVIOUS_GENES_TO_KEEP = 100;

    public static class Builder
    {
        private ChromosomeFactory chromosomeFactory;
        private PopulationModel populationModel;
        private CrossoverOperator crossoverOperator;
        private MutationOperator mutationOperator;
        private RandomGenerator rng;

        public GeneticSelectionOperator.Builder crossoverOperator(CrossoverOperator crossoverOperator) {
            this.crossoverOperator = crossoverOperator;
            return this;
        }

        public GeneticSelectionOperator.Builder mutationOperator(MutationOperator mutationOperator) {
            this.mutationOperator = mutationOperator;
            return this;
        }

        public GeneticSelectionOperator.Builder randomGenerator(RandomGenerator rng) {
            this.rng = rng;
            return this;
        }

        public GeneticSelectionOperator build() {
            if(crossoverOperator == null) {
                crossoverOperator = new SinglePointCrossover.Builder().build();
            }

            if(mutationOperator == null) {
                mutationOperator = new RandomMutationOperator.Builder().build();
            }

            if(rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }

            return new GeneticSelectionOperator(crossoverOperator, mutationOperator, rng);
        }
    }

    private final CrossoverOperator crossoverOperator;
    private final MutationOperator mutationOperator;
    private final RandomGenerator rng;
    private double[][] previousGenes = new double[PREVIOUS_GENES_TO_KEEP][];
    private int previousGenesIdx = 0;


    public GeneticSelectionOperator(CrossoverOperator crossoverOperator, MutationOperator mutationOperator, RandomGenerator rng) {
        this.crossoverOperator = crossoverOperator;
        this.mutationOperator = mutationOperator;
        this.rng = rng;
    }

    @Override
    public void initializeInstance(PopulationModel populationModel, ChromosomeFactory chromosomeFactory) {
        super.initializeInstance(populationModel, chromosomeFactory);
        crossoverOperator.initializeInstance(populationModel);
    }

    @Override
    public double[] buildNextGenes() {
        double[] result;

        do {
            if (populationModel.isReadyToBreed()) {
                result = buildOffspring();
            } else {
                result = buildRandomGenes();
            }
        } while (hasAlreadyBeenTried(result));

        previousGenes[previousGenesIdx] = result;
        previousGenesIdx = ++previousGenesIdx % previousGenes.length;

        return result;
    }

    private boolean hasAlreadyBeenTried(double[] genes) {
        for(int i = 0; i < previousGenes.length; ++i) {
            double[] current = previousGenes[i];
            if(current != null && Arrays.equals(current, genes)) {
                return true;
            }
        }

        return false;
    }

    private double[] buildOffspring() {
        double[] offspringValues;

        boolean hasModification = false;
        do {
            CrossoverResult crossoverResult = crossoverOperator.crossover();
            offspringValues = crossoverResult.genes;
            hasModification |= crossoverResult.hasModification;
            hasModification |= mutationOperator.mutate(offspringValues);
        } while (!hasModification);
        return offspringValues;
    }

    private double[] buildRandomGenes() {
        double[] randomValues = new double[chromosomeFactory.getChromosomeLength()];
        for (int i = 0; i < randomValues.length; ++i) {
            randomValues[i] = rng.nextDouble();
        }

        return randomValues;
    }

}
