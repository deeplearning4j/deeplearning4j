package org.arbiter.optimize.runner;

import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.saving.ResultReference;

import java.util.List;

public interface IOptimizationRunner<T,M,A> {

    void execute();

    /** Total number of candidates: created (scheduled), completed and failed */
    int numCandidatesScheduled();

    int numCandidatesCompleted();

    int numCandidatesFailed();

    /** Best score found so far */
    double bestScore();

    /** Time that the best score was found at, or 0 if no jobs have completed successfully */
    long bestScoreTime();

    List<ResultReference<T,M,A>> getResults();

}
