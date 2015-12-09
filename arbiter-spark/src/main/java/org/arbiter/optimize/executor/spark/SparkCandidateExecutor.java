package org.arbiter.optimize.executor.spark;

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
import java.util.concurrent.Future;


@AllArgsConstructor
public class SparkCandidateExecutor<T,M,D> implements CandidateExecutor<T,M,D> {

    private JavaSparkContext sparkContext;
    private TaskCreator<T,M,D> taskCreator;


    @Override
    public Future<OptimizationResult<T, M>> execute(Candidate<T> candidate, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction) {
        return execute(Collections.singletonList(candidate),dataProvider,scoreFunction).get(0);
    }

    @Override
    public List<Future<OptimizationResult<T, M>>> execute(List<Candidate<T>> candidates, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction) {
        List<CandidateDataPair<T,D>> list = new ArrayList<>(candidates.size());
        for(Candidate<T> candidate : candidates ){
            list.add(new CandidateDataPair<>(candidate, dataProvider));
        }
        JavaRDD<CandidateDataPair<T,D>> rdd = sparkContext.parallelize(list);

        JavaRDD<OptimizationResult<T,M>> results = rdd.map(new ExecuteFunction<T, M, D>());

        JavaFutureAction<List<OptimizationResult<T,M>>> out = results.collectAsync();

        //Problem: collectAsync allows us to get JavaFutureAction<List<OptimizationResult<T,M>>>
        // which is equivalent to Future<List<OptimizationResult<T,M>>> (as JavaFutureAction IS a Future)
        // but what we REALLY want is List<Future<OptimizationResult<T,M>>>
        // Can probably handle this with a layer of indirection:
        //An additional complication here: I don't think order (of output) is guaranteed here.
        // Might need to key by integer index in the original (input) list
        List<OptimizationResult<T,M>> collected;
        try {
            collected = out.get();
        } catch(Exception e){
            throw new RuntimeException(e);
        }

        return null;
    }

    @AllArgsConstructor
    private class Job {
        private final Candidate<T> candidate;
        private final DataProvider<D> dataProvider;
    }
}
