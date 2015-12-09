package org.arbiter.optimize.runner;

public interface IOptimizationRunner {

    void execute();

    /** Total number of candidates: created (scheduled), completed and failed */
    int numCandidatesScheduled();

    int numCandidatesCompleted();

    int numCandidatesFailed();

    /** Best score found so far */
    double bestScore();

    /** Time that the best score was found at, or 0 if no jobs have completed successfully */
    long bestScoreTime();

}
