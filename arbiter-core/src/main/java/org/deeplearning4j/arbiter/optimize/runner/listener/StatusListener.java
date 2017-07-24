package org.deeplearning4j.arbiter.optimize.runner.listener;

import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;

/**
 * The status Listener interface is used to inspect/track the status of execution, both for individual candidates,
 * and for the optimisation runner overall.
 *
 * @author Alex Black
 */
public interface StatusListener {

    /** Called when optimization runner starts execution */
    void onInitialization(IOptimizationRunner runner);

    /** Called when optimization runner terminates */
    void onShutdown(IOptimizationRunner runner);

    /** Called when any of the summary stats change, for the optimization runner:
     * number scheduled, number completed, number failed, best score, etc. */
    void onRunnerStatusChange(IOptimizationRunner runner);

    /**
     * Called when the status of the candidate is change. For example created, completed, failed.
     *
     * @param candidateInfo Candidate information
     * @param runner        Optimisation runner calling this method
     * @param result        Optimisation result. Maybe null.
     */
    void onCandidateStatusChange(CandidateInfo candidateInfo, IOptimizationRunner runner, OptimizationResult result);

    /**
     *  This method may be called by tasks as they are executing. The intent of this method is to report partial results,
     *  such as different stages of learning, or scores/evaluations so far
     *
     * @param candidateInfo Candidate information
     * @param candidate     Current candidate value/configuration
     * @param iteration     Current iteration number
     */
    void onCandidateIteration(CandidateInfo candidateInfo, Object candidate, int iteration);

}
