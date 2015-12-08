package org.arbiter.optimize.executor.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.executor.spark.CandidateDataPair;

public class ExecuteFunction<T,M,D> implements Function<CandidateDataPair<T,D>,OptimizationResult<T,M>> {

    @Override
    public OptimizationResult<T, M> call(CandidateDataPair<T, D> tdCandidateDataPair) throws Exception {
        return null;
    }
}
