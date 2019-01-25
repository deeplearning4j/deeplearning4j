package org.deeplearning4j.arbiter.optimize.generator.genetic.culling;

public class LeastFitCullOperation extends RatioCullOperation {

    public LeastFitCullOperation() {
        super();
    }

    public LeastFitCullOperation(double cullRatio) {
        super(cullRatio);
    }

    @Override
    public void cullPopulation() {
        while (population.size() > culledSize) {
            population.remove(population.size() - 1);
        }
    }
}
