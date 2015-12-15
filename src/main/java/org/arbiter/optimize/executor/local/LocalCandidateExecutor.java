package org.arbiter.optimize.executor.local;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.executor.CandidateExecutor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;


public class LocalCandidateExecutor<T,M,D> implements CandidateExecutor<T,M,D> {

    private TaskCreator<T,M,D> taskCreator;
    private ListeningExecutorService executor;
    private final int nThreads;

    public LocalCandidateExecutor(TaskCreator<T,M,D> taskCreator){
        this(taskCreator,1);
    }

    public LocalCandidateExecutor(TaskCreator<T,M,D> taskCreator, int nThreads){
        this.taskCreator = taskCreator;
        this.nThreads = nThreads;

        ExecutorService exec = Executors.newFixedThreadPool(nThreads, new ThreadFactory() {
            private AtomicLong counter = new AtomicLong(0);
            @Override
            public Thread newThread(Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);
                t.setDaemon(true);
                t.setName("LocalCandidateExecutor-"+counter.getAndIncrement());
                return t;
            }
        });
        executor = MoreExecutors.listeningDecorator(exec);
    }


    @Override
    public ListenableFuture<OptimizationResult<T, M>> execute(Candidate<T> candidate, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction ) {
        Callable<OptimizationResult<T,M>> task = taskCreator.create(candidate,dataProvider,scoreFunction);
        return executor.submit(task);
    }

    @Override
    public List<ListenableFuture<OptimizationResult<T, M>>> execute(List<Candidate<T>> candidates, DataProvider<D> dataProvider, ScoreFunction<M,D> scoreFunction ) {
        List<ListenableFuture<OptimizationResult<T,M>>> list = new ArrayList<>(candidates.size());
        for(Candidate<T> candidate : candidates){
            Callable<OptimizationResult<T,M>> task = taskCreator.create(candidate, dataProvider, scoreFunction);
            list.add(executor.submit(task));
        }
        return list;
    }

    @Override
    public int maxConcurrentTasks() {
        return nThreads;
    }

    @Override
    public void shutdown() {
        executor.shutdownNow();
    }

    @AllArgsConstructor
    private class Job {
        private Candidate<T> candidate;
        private DataProvider<D> dataProvider;
    }
}
