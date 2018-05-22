package org.deeplearning4j.arbiter.optimize.runner.listener.impl;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;

/**
 * Created by Alex on 20/07/2017.
 */
@Slf4j
public class LoggingStatusListener implements StatusListener {


    @Override
    public void onInitialization(IOptimizationRunner runner) {
        log.info("Optimization runner: initialized");
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {
        log.info("Optimization runner: shut down");
    }

    @Override
    public void onRunnerStatusChange(IOptimizationRunner runner) {
        log.info("Optimization runner: status change");
    }

    @Override
    public void onCandidateStatusChange(CandidateInfo candidateInfo, IOptimizationRunner runner,
                    OptimizationResult result) {
        log.info("Candidate status change: {}", candidateInfo);
    }

    @Override
    public void onCandidateIteration(CandidateInfo candidateInfo, Object candidate, int iteration) {
        log.info("Candidate iteration #{} - {}", iteration, candidate);
    }
}
