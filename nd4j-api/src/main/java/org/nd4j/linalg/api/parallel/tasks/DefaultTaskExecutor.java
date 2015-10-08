package org.nd4j.linalg.api.parallel.tasks;


import java.util.concurrent.*;

public class DefaultTaskExecutor implements TaskExecutor {

    private static DefaultTaskExecutor instance;
    private ExecutorService executorService;

    static {
        instance = new DefaultTaskExecutor();
    }

    public static DefaultTaskExecutor getInstance(){
        return instance;
    }

    public DefaultTaskExecutor(){
        int nThreads = Runtime.getRuntime().availableProcessors();
        executorService = new ThreadPoolExecutor(nThreads,nThreads,60, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>() );
    }

    @Override
    public <V> Future<V> executeAsync(Task<V> task) {
        Future<V> f = new FutureTask<>(task);
        executorService.submit(task);
        return f;
    }
}
