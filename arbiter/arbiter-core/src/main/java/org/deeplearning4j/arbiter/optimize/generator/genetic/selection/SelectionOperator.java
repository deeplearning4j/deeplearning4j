package org.deeplearning4j.arbiter.optimize.generator.genetic.selection;

import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

public abstract class SelectionOperator {
    protected PopulationModel populationModel;
    protected ChromosomeFactory chromosomeFactory;

    public void initializeInstance(PopulationModel populationModel, ChromosomeFactory chromosomeFactory) {

        this.populationModel = populationModel;
        this.chromosomeFactory = chromosomeFactory;
    }
    public abstract double[] buildNextGenes();
}
