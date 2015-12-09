package org.arbiter.optimize.executor;

import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;

import java.util.List;
import java.util.concurrent.Future;

public interface CandidateExecutor<T,M,D> {

    Future<OptimizationResult<T,M>> execute(Candidate<T> candidate, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction );

    List<Future<OptimizationResult<T,M>>> execute(List<Candidate<T>> candidates, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction );



}
