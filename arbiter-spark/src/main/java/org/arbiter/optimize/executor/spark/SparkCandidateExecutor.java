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
package org.arbiter.optimize.executor.spark;

import com.google.common.util.concurrent.JdkFutureAdapters;
import com.google.common.util.concurrent.ListenableFuture;
import lombok.AllArgsConstructor;
import org.apache.spark.api.java.JavaFutureAction;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.executor.CandidateExecutor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;


@AllArgsConstructor
public class SparkCandidateExecutor<C,M,D,A> implements CandidateExecutor<C,M,D,A> {

    private JavaSparkContext sparkContext;
    private TaskCreator<C,M,D,A> taskCreator;


    @Override
    public ListenableFuture<OptimizationResult<C, M, A>> execute(Candidate<C> candidate, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction) {
        return execute(Collections.singletonList(candidate),dataProvider,scoreFunction).get(0);
    }

    @Override
    public List<ListenableFuture<OptimizationResult<C, M, A>>> execute(List<Candidate<C>> candidates, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction) {
        List<ListenableFuture<OptimizationResult<C,M,A>>> list = new ArrayList<>(candidates.size());
        for(Candidate<C> candidate : candidates) {
            CandidateDataScoreTuple<C,D,M> tuple =
                    new CandidateDataScoreTuple<>();
            tuple.setCandidate(candidate);
            tuple.setDataProvider(dataProvider);
            tuple.setScoreFunction(scoreFunction);
            List<CandidateDataScoreTuple<C,D,M>> singleList = new ArrayList<>();
            singleList.add(tuple);
            JavaRDD<CandidateDataScoreTuple<C,D,M>> rdd = sparkContext.parallelize(singleList);
            JavaRDD<OptimizationResult<C,M,A>> results = rdd.map(new Function<CandidateDataScoreTuple<C, D, M>, OptimizationResult<C, M, A>>() {
                @Override
                public OptimizationResult<C, M, A> call(CandidateDataScoreTuple<C, D, M> cdmCandidateDataScoreTuple) throws Exception {
                    return null;
                }
            });

            JavaFutureAction<List<OptimizationResult<C,M,A>>> out = results.collectAsync();
            Future<OptimizationResult<C,M,A>> f = new FutureListAdapter<>(out);
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
        private  Candidate<C> candidate;
        private  DataProvider<D> dataProvider;
    }

    @AllArgsConstructor
    private class FutureListAdapter<C,M,A> implements Future<OptimizationResult<C,M,A>> {
        private JavaFutureAction<List<OptimizationResult<C,M,A>>> futureAction;

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
        public OptimizationResult<C, M, A> get() throws InterruptedException, ExecutionException {
            return futureAction.get().get(0);
        }

        @Override
        public OptimizationResult<C, M, A> get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
            return futureAction.get(timeout,unit).get(0);
        }
    }
}
