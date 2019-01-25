package org.deeplearning4j.arbiter.optimize.generator.genetic.population;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;

import java.util.List;

public interface PopulationListener {
    void onChanged(List<Chromosome> population);
}
