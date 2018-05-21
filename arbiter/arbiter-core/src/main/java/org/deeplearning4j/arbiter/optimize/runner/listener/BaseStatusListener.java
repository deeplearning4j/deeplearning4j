package org.deeplearning4j.arbiter.optimize.runner.listener;

import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;

/**
 * BaseStatusListener: implements all methods of {@link StatusListener} as no-op.
 * Users can extend this and override only the methods actually required
 *
 * @author Alex Black
 */
public abstract class BaseStatusListener implements StatusListener{
    @Override
    public void onInitialization(IOptimizationRunner runner) {
        //No op
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {
        //No op
    }

    @Override
    public void onRunnerStatusChange(IOptimizationRunner runner) {
        //No op
    }

    @Override
    public void onCandidateStatusChange(CandidateInfo candidateInfo, IOptimizationRunner runner, OptimizationResult result) {
        //No op
    }

    @Override
    public void onCandidateIteration(CandidateInfo candidateInfo, Object candidate, int iteration) {
        //No op
    }
}
