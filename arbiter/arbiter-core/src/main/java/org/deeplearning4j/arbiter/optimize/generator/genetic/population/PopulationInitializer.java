package org.deeplearning4j.arbiter.optimize.generator.genetic.population;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;

import java.util.List;

public interface PopulationInitializer {
    List<Chromosome> getInitializedPopulation(int size);
}
