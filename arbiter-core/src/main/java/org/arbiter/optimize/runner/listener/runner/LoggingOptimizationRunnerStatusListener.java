package org.arbiter.optimize.runner.listener.runner;

import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.runner.IOptimizationRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by Alex on 21/12/2015.
 */
public class LoggingOptimizationRunnerStatusListener implements OptimizationRunnerStatusListener {

    private static final Logger log = LoggerFactory.getLogger(LoggingOptimizationRunnerStatusListener.class);

    @Override
    public void onInitialization(IOptimizationRunner runner) {
        log.info("Optimization runner: Initialized.");
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {
        log.info("Optimization runner: shutting down.");
    }

    @Override
    public void onStatusChange(IOptimizationRunner runner) {
        log.info("OptimizationRunner - status change"); //TODO
    }

    @Override
    public void onCompletion(OptimizationResult<?, ?, ?> result) {
        log.info("Optimization runner: task complete. Index = {}, score = {}",result.getIndex(), result.getScore());
    }
}
