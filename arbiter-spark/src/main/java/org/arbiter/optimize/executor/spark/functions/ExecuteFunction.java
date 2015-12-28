package org.arbiter.optimize.executor.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.executor.spark.CandidateDataScoreTuple;

public class ExecuteFunction<T,M,D,A> implements Function<CandidateDataScoreTuple<T,M,D>,OptimizationResult<T,M,A>> {

    @Override
    public OptimizationResult<T, M, A> call(CandidateDataScoreTuple<T, M, D> input) throws Exception {
        //TODO

        return null;
    }
}
