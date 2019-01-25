package org.deeplearning4j.arbiter.optimize.generator.genetic.culling;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

import java.util.List;

public abstract class RatioCullOperation implements CullOperation {
    public static final double DEFAULT_CULL_RATIO = 1.0 / 3.0;
    protected int culledSize;
    protected List<Chromosome> population;

    public RatioCullOperation(double cullRatio) {
        this.cullRatio = cullRatio;
    }

    public RatioCullOperation() {
        this(DEFAULT_CULL_RATIO);
    }

    protected double cullRatio;
    public double getCullRatio() {
        return cullRatio;
    }
    public void setCullRatio(double cullRatio) {
        this.cullRatio = cullRatio;
    }

    public void initializeInstance(PopulationModel populationModel) {
        this.population = populationModel.population;
        culledSize = (int)(populationModel.populationSize * (1.0 - cullRatio) + 0.5);
    }

    @Override
    public int getCulledSize() {
        return culledSize;
    }

}
