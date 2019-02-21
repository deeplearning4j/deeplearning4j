package org.deeplearning4j.arbiter.optimize.genetic;

import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.MutationOperator;

public class TestMutationOperator implements MutationOperator {

    private final boolean[] results;
    private int resultIdx = 0;

    public TestMutationOperator(boolean[] results) {
        this.results = results;
    }

    @Override
    public boolean mutate(double[] genes) {
        return results[resultIdx++];
    }
}
