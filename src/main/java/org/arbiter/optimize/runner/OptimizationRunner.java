package org.arbiter.optimize.runner;

import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.executor.CandidateExecutor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;


public class OptimizationRunner<T, M, D> {

    private OptimizationConfiguration<T, M, D> config;
    private CandidateExecutor<T, M, D> executor;
    private List<Future<OptimizationResult<T, M>>> futures = new ArrayList<>();  //TODO: use threadsafe list?


    public OptimizationRunner(OptimizationConfiguration<T, M, D> config, CandidateExecutor<T, M, D> executor) {
        this.config = config;
        this.executor = executor;
    }

    public void execute() {


    }

}
