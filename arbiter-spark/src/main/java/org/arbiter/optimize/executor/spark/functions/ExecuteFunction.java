package org.arbiter.optimize.executor.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.executor.spark.CandidateDataScoreTuple;

public class ExecuteFunction<T,M,D> implements Function<CandidateDataScoreTuple<T,M,D>,OptimizationResult<T,M>> {

    @Override
    public OptimizationResult<T, M> call(CandidateDataScoreTuple<T, M, D> input) throws Exception {
        //TODO

        return null;
    }
}
