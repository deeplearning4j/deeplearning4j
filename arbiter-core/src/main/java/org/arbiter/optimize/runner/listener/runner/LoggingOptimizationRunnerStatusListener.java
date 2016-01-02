/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
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
