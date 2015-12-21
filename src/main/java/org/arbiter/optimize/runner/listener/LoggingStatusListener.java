package org.arbiter.optimize.runner.listener;

import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.runner.IOptimizationRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by Alex on 21/12/2015.
 */
public class LoggingStatusListener implements StatusListener {

    private static final Logger log = LoggerFactory.getLogger(LoggingStatusListener.class);

    @Override
    public void onInitialization(IOptimizationRunner runner) {
        log.info("Optimization runner: Initialized.");
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {
        log.info("Optimization runner: shutting down.");
    }

    @Override
    public void onCompletion(OptimizationResult<?, ?, ?> result) {
        log.info("Optimization runner: task complete. Index = {}, score = {}",result.getIndex(), result.getScore());
    }
}
