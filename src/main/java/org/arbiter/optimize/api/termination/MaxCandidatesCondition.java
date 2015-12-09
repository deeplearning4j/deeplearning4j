package org.arbiter.optimize.api.termination;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.runner.IOptimizationRunner;

@AllArgsConstructor
public class MaxCandidatesCondition implements TerminationCondition {

    private final int maxCandidates;

    @Override
    public void initialize(IOptimizationRunner optimizationRunner) {
        //No op
    }

    @Override
    public boolean terminate(IOptimizationRunner optimizationRunner) {
        return optimizationRunner.numCandidatesScheduled() >= maxCandidates;
    }

    @Override
    public String toString(){
        return "MaxCandidatesCondition("+maxCandidates+")";
    }
}
