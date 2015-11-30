package org.nd4j.linalg.api.parallel.tasks;


import org.nd4j.linalg.executors.ExecutorServiceProvider;

import java.util.concurrent.*;

/**
 * Default TaskExecutor based on a
 * (a) ForkJoinPool (for CPU tasks that are ForkJoin RecursiveTask/RecursiveAction, for example)
 *
 * (b) ThreadPoolExecutor (for all other tasks)
 * number of threads set to the number of processor (cores) by default, as per the
 */
public class DefaultTaskExecutor implements TaskExecutor {

    private static DefaultTaskExecutor instance = new DefaultTaskExecutor();
    private ExecutorService executorService;
    private ForkJoinPool forkJoinPool;

    public static DefaultTaskExecutor getInstance(){
        return instance;
    }

    public DefaultTaskExecutor(){

    }

    @Override
    public <V> Future<V> executeAsync(Task<V> task) {
        if(task instanceof ForkJoinTask ) {
            if(forkJoinPool == null) forkJoinPool = ExecutorServiceProvider.getForkJoinPool();
            forkJoinPool.execute((ForkJoinTask<?>)task);
            return (Future<V>)task;
        } else {
            if(executorService == null) executorService = ExecutorServiceProvider.getExecutorService();
            return executorService.submit(task);
        }
    }
}
