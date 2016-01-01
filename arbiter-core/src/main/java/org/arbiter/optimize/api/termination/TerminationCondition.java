package org.arbiter.optimize.api.termination;


import org.arbiter.optimize.runner.IOptimizationRunner;

/** Global termination condition */
public interface TerminationCondition {

    void initialize(IOptimizationRunner optimizationRunner);

    boolean terminate(IOptimizationRunner optimizationRunner);

}
