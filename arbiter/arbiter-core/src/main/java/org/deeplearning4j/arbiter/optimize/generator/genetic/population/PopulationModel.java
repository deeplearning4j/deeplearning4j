package org.deeplearning4j.arbiter.optimize.generator.genetic.population;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.CullOperation;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.LeastFitCullOperation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class PopulationModel {
    private static final int DEFAULT_POPULATION_SIZE = 30;

    public class MaximizeScoreComparator implements Comparator<Chromosome> {
        @Override
        public int compare(Chromosome lhs, Chromosome rhs) {
            if (lhs.fitness < rhs.fitness)
                return 1;
            else if (rhs.fitness < lhs.fitness)
                return -1;
            return 0;
        }
    }

    public class MinimizeScoreComparator implements Comparator<Chromosome> {

        @Override
        public int compare(Chromosome lhs, Chromosome rhs) {
            if (lhs.fitness < rhs.fitness)
                return -1;
            else if (rhs.fitness < lhs.fitness)
                return 1;
            return 0;
        }
    }

    public static class Builder {
        private int populationSize = DEFAULT_POPULATION_SIZE;
        private PopulationInitializer populationInitializer;
        private CullOperation cullOperation;

        public Builder(PopulationInitializer populationInitializer) {
            this.populationInitializer = populationInitializer;
        }

        public PopulationModel.Builder populationSize(int size) {
            populationSize = size;
            return this;
        }

        public PopulationModel.Builder cullingOperation(CullOperation cullOperation) {
            this.cullOperation = cullOperation;
            return this;
        }

        public PopulationModel build() {
            if(cullOperation == null) {
                cullOperation = new LeastFitCullOperation();
            }

            return new PopulationModel(this);
        }

    }

    public final int populationSize;
    public final List<Chromosome> population;

    private final PopulationInitializer populationInitializer;
    private final CullOperation cullOperation;
    private final List<PopulationListener> populationListeners = new ArrayList<>();
    private Comparator<Chromosome> chromosomeComparator;

    public PopulationModel(PopulationModel.Builder builder) {
        populationSize = builder.populationSize;
        population = new ArrayList<>(builder.populationSize);
        populationInitializer = builder.populationInitializer;

        List<Chromosome> initializedPopulation = populationInitializer.getInitializedPopulation(populationSize);
        population.clear();
        population.addAll(initializedPopulation);

        cullOperation = builder.cullOperation;
        cullOperation.initializeInstance(this);
    }

    public void initializeInstance(boolean minimizeScore) {
        chromosomeComparator = minimizeScore ? new MinimizeScoreComparator() : new MaximizeScoreComparator();
    }

    public void addListener(PopulationListener listener) {
        populationListeners.add(listener);
    }

    public void add(Chromosome element) {
        if(population.size() == populationSize) {
            cullOperation.cullPopulation();
        }

        population.add(element);

        Collections.sort(population, chromosomeComparator);

        triggerPopulationChangedListeners(population);
    }

    public boolean isReadyToBreed() {
        return population.size() >= cullOperation.getCulledSize();
    }

    private void triggerPopulationChangedListeners(List<Chromosome> population) {
        for(PopulationListener listener : populationListeners) {
            listener.onChanged(population);
        }
    }
}
