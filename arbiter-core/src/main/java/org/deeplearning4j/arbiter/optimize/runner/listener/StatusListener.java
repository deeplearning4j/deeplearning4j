package org.deeplearning4j.arbiter.optimize.runner.listener;

import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;

/**
 * Created by Alex on 20/07/2017.
 */
public interface StatusListener {

    /** Called when optimization runner starts execution */
    void onInitialization(IOptimizationRunner runner);

    /** Called when optimization runner terminates */
    void onShutdown(IOptimizationRunner runner);

    /** Called when any of the summary stats change, for the optimization runner:
     * number scheduled, number completed, number failed, best score, etc. */
    void onRunnerStatusChange(StatusChangeType statusChangeType, IOptimizationRunner runner);

    void onCandidateStatusChange(CandidateInfo candidateInfo, IOptimizationRunner runner,
                    OptimizationResult<?, ?, ?> result);

    /**
     *  this method may be called by tasks as they are executing. The intent of this method is to report partial results,
     *  such as different stages of learning, or scores/evaluations so far
     *
     * @param candidate
     * @param iteration
     */
    void onCandidateIteration(Object candidate, int iteration);

}
