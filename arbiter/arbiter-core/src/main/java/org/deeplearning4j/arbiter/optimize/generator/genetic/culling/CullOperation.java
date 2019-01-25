package org.deeplearning4j.arbiter.optimize.generator.genetic.culling;

import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

public interface CullOperation {
    void initializeInstance(PopulationModel populationModel);
    void cullPopulation();
    int getCulledSize();
}
