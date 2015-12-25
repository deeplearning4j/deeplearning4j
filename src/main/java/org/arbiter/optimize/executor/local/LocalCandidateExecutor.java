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
import org.arbiter.optimize.runner.listener.candidate.UICandidateStatusListenerImpl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;


public class LocalCandidateExecutor<T, M, D, A> implements CandidateExecutor<T, M, D, A> {

    private TaskCreator<T, M, D, A> taskCreator;
    private ListeningExecutorService executor;
    private final int nThreads;

    public LocalCandidateExecutor(TaskCreator<T, M, D, A> taskCreator) {
        this(taskCreator, 1);
    }

    public LocalCandidateExecutor(TaskCreator<T, M, D, A> taskCreator, int nThreads) {
        this.taskCreator = taskCreator;
        this.nThreads = nThreads;

        ExecutorService exec = Executors.newFixedThreadPool(nThreads, new ThreadFactory() {
            private AtomicLong counter = new AtomicLong(0);

            @Override
            public Thread newThread(Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);
                t.setDaemon(true);
                t.setName("LocalCandidateExecutor-" + counter.getAndIncrement());
                return t;
            }
        });
        executor = MoreExecutors.listeningDecorator(exec);
    }


    @Override
    public ListenableFuture<OptimizationResult<T, M, A>> execute(Candidate<T> candidate, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction) {
        return execute(Collections.singletonList(candidate),dataProvider,scoreFunction).get(0);
    }

    @Override
    public List<ListenableFuture<OptimizationResult<T, M, A>>> execute(List<Candidate<T>> candidates, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction) {
        List<ListenableFuture<OptimizationResult<T, M, A>>> list = new ArrayList<>(candidates.size());
        for (Candidate<T> candidate : candidates) {
            Callable<OptimizationResult<T, M, A>> task = taskCreator.create(candidate, dataProvider, scoreFunction,
                    new UICandidateStatusListenerImpl(candidate.getIndex()));
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
