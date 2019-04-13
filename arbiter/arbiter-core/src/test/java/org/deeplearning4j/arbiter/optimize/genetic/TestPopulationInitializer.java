package org.deeplearning4j.arbiter.optimize.genetic;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;

import java.util.ArrayList;
import java.util.List;

public class TestPopulationInitializer implements PopulationInitializer {
    @Override
    public List<Chromosome> getInitializedPopulation(int size) {
        return new ArrayList<>();
    }
}
