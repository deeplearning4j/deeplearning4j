package org.deeplearning4j.nn.api;

/**
 * Optimization algorithm to use
 * @author Adam Gibson
 *
 */
public enum OptimizationAlgorithm {
    GRADIENT_DESCENT,
    CONJUGATE_GRADIENT,
    HESSIAN_FREE,
    LBFGS,
    ITERATION_GRADIENT_DESCENT
}
