package org.arbiter.optimize.executor.spark;

import com.google.common.util.concurrent.JdkFutureAdapters;
import com.google.common.util.concurrent.ListenableFuture;
import lombok.AllArgsConstructor;
import org.apache.spark.api.java.JavaFutureAction;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.executor.CandidateExecutor;
import org.arbiter.optimize.executor.spark.functions.ExecuteFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;


@AllArgsConstructor
public class SparkCandidateExecutor<T,M,D> implements CandidateExecutor<T,M,D> {

    private JavaSparkContext sparkContext;
    private TaskCreator<T,M,D> taskCreator;


    @Override
    public ListenableFuture<OptimizationResult<T, M>> execute(Candidate<T> candidate, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction) {
        return execute(Collections.singletonList(candidate),dataProvider,scoreFunction).get(0);
    }

    @Override
    public List<ListenableFuture<OptimizationResult<T, M>>> execute(List<Candidate<T>> candidates, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction) {
        List<ListenableFuture<OptimizationResult<T,M>>> list = new ArrayList<>(candidates.size());
        for(Candidate<T> candidate : candidates ){
            JavaRDD<CandidateDataScoreTuple<T,M,D>> rdd = sparkContext.parallelize(Collections.singletonList(
                    new CandidateDataScoreTuple<T, M, D>(candidate, dataProvider, scoreFunction)));

            JavaRDD<OptimizationResult<T,M>> results = rdd.map(new ExecuteFunction<T, M, D>());

            JavaFutureAction<List<OptimizationResult<T,M>>> out = results.collectAsync();
            Future<OptimizationResult<T,M>> f = new FutureListAdapter(out);
            list.add(JdkFutureAdapters.listenInPoolThread(f));
        }

        return list;
    }

    @Override
    public int maxConcurrentTasks() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void shutdown() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @AllArgsConstructor
    private class Job {
        private final Candidate<T> candidate;
        private final DataProvider<D> dataProvider;
    }

    @AllArgsConstructor
    private class FutureListAdapter implements Future<OptimizationResult<T,M>>{
        private JavaFutureAction<List<OptimizationResult<T,M>>> futureAction;

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            return futureAction.cancel(mayInterruptIfRunning);
        }

        @Override
        public boolean isCancelled() {
            return futureAction.isCancelled();
        }

        @Override
        public boolean isDone() {
            return futureAction.isDone();
        }

        @Override
        public OptimizationResult<T, M> get() throws InterruptedException, ExecutionException {
            return futureAction.get().get(0);
        }

        @Override
        public OptimizationResult<T, M> get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
            return futureAction.get(timeout,unit).get(0);
        }
    }
}
