package org.deeplearning4j.arbiter.optimize.generator.genetic.mutation;

public interface MutationOperator {
    boolean mutate(double[] genes);
}
