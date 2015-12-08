package org.arbiter.optimize.executor.local;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.executor.CandidateExecutor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;


public class LocalCandidateExecutor<T,M,D> implements CandidateExecutor<T,M,D> {

    private TaskCreator<T,M> taskCreator;
    private int nThreads;
    private ExecutorService executor;

    public LocalCandidateExecutor(TaskCreator<T,M> taskCreator){
        this(taskCreator,1);
    }

    public LocalCandidateExecutor(TaskCreator<T,M> taskCreator, int nThreads){
        this.nThreads = nThreads;

        executor = Executors.newFixedThreadPool(nThreads, new ThreadFactory() {
            private AtomicLong counter = new AtomicLong(0);
            @Override
            public Thread newThread(Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);
                t.setDaemon(true);
                t.setName("LocalCandidateExecutor-"+counter.getAndIncrement());
                return t;
            }
        });
    }


    @Override
    public Future<OptimizationResult<T, M>> execute(Candidate<T> candidate, DataProvider<D> dataProvider) {
        Callable<OptimizationResult<T,M>> task = taskCreator.create(candidate,dataProvider);
        return executor.submit(task);

    }

    @Override
    public List<Future<OptimizationResult<T, M>>> execute(List<Candidate<T>> candidates, DataProvider<D> dataProvider) {
        List<Future<OptimizationResult<T,M>>> list = new ArrayList<>(candidates.size());
        for(Candidate<T> candidate : candidates){
            Callable<OptimizationResult<T,M>> task = taskCreator.create(candidate,dataProvider);
            list.add(executor.submit(task));
        }
        return list;
    }

    @AllArgsConstructor
    private class Job {
        private Candidate<T> candidate;
        private DataProvider<D> dataProvider;
    }
}
