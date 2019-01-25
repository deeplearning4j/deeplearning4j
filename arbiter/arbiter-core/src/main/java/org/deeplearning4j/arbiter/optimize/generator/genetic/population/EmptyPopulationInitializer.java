package org.deeplearning4j.arbiter.optimize.generator.genetic.population;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;

import java.util.ArrayList;
import java.util.List;

public class EmptyPopulationInitializer implements PopulationInitializer {

    @Override
    public List<Chromosome> getInitializedPopulation(int size)
    {
        return new ArrayList<>(size);
    }
}
