package org.arbiter.optimize.executor;

import com.google.common.util.concurrent.ListenableFuture;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;

import java.util.List;
import java.util.concurrent.Future;

public interface CandidateExecutor<T,M,D,A> {

    ListenableFuture<OptimizationResult<T,M,A>> execute(Candidate<T> candidate, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction);

    List<ListenableFuture<OptimizationResult<T,M,A>>> execute(List<Candidate<T>> candidates, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction);

    int maxConcurrentTasks();

    /** Shut down the executor, and immediately cancel all remaining tasks */
    void shutdown();

}
