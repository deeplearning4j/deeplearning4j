package org.arbiter.optimize.runner;

import org.arbiter.optimize.api.saving.ResultReference;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.runner.listener.runner.OptimizationRunnerStatusListener;

import java.util.List;

public interface IOptimizationRunner<C,M,A> {

    void execute();

    /** Total number of candidates: created (scheduled), completed and failed */
    int numCandidatesTotal();

    int numCandidatesCompleted();

    int numCandidatesFailed();

    /** Number of candidates running or queued */
    int numCandidatesQueued();

    /** Best score found so far */
    double bestScore();

    /** Time that the best score was found at, or 0 if no jobs have completed successfully */
    long bestScoreTime();

    /** Index of the best scoring candidate, or -1 if no candidate has scored yet*/
    int bestScoreCandidateIndex();

    List<ResultReference<C,M,A>> getResults();

    OptimizationConfiguration<C,M,?,A> getConfiguration();

    void addListeners(OptimizationRunnerStatusListener... listeners);

    void removeListeners(OptimizationRunnerStatusListener... listeners);

    void removeAllListeners();

    List<CandidateStatus> getCandidateStatus();

}
