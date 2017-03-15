package org.deeplearning4j.arbiter.optimize.runner;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.TaskCreator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.listener.candidate.UICandidateStatusListenerImpl;
import org.deeplearning4j.arbiter.optimize.ui.ArbiterUIServer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicLong;

/**
 * LocalOptimizationRunner: execute hyperparameter optimization
 * locally (on current machine, in current JVM).
 *
 * @param <C> Type for candidate configurations
 * @param <M> Type of trained model
 * @param <D> Type of data used for hyperparameter optimization
 * @param <A> Type of any additional evaluation/results
 * @author Alex Black
 */
public class LocalOptimizationRunner<C, M, D, A> extends BaseOptimizationRunner<C, M, D, A> {

    public static final int DEFAULT_MAX_CONCURRENT_TASKS = 1;

    private final int maxConcurrentTasks;

    private TaskCreator<C, M, D, A> taskCreator;
    private ListeningExecutorService executor;

    public LocalOptimizationRunner(OptimizationConfiguration<C, M, D, A> config, TaskCreator<C, M, D, A> taskCreator) {
        this(DEFAULT_MAX_CONCURRENT_TASKS, config, taskCreator);
    }

    public LocalOptimizationRunner(int maxConcurrentTasks, OptimizationConfiguration<C, M, D, A> config, TaskCreator<C, M, D, A> taskCreator) {
        super(config);
        if (maxConcurrentTasks <= 0)
            throw new IllegalArgumentException("maxConcurrentTasks must be > 0 (got: " + maxConcurrentTasks + ")");
        this.maxConcurrentTasks = maxConcurrentTasks;
        this.taskCreator = taskCreator;

        ExecutorService exec = Executors.newFixedThreadPool(maxConcurrentTasks, new ThreadFactory() {
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

        init();
    }

    @Override
    protected int maxConcurrentTasks() {
        return maxConcurrentTasks;
    }

    @Override
    protected ListenableFuture<OptimizationResult<C, M, A>> execute(Candidate<C> candidate, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction) {
        return execute(Collections.singletonList(candidate), dataProvider, scoreFunction).get(0);
    }

    @Override
    protected List<ListenableFuture<OptimizationResult<C, M, A>>> execute(List<Candidate<C>> candidates, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction) {
        List<ListenableFuture<OptimizationResult<C, M, A>>> list = new ArrayList<>(candidates.size());
        for (Candidate<C> candidate : candidates) {
            Callable<OptimizationResult<C, M, A>> task = taskCreator.create(candidate, dataProvider, scoreFunction,
                    (ArbiterUIServer.isRunning() ? new UICandidateStatusListenerImpl(candidate.getIndex()) : null));
            list.add(executor.submit(task));
        }
        return list;
    }

    @Override
    protected void shutdown() {
        executor.shutdownNow();
    }
}
