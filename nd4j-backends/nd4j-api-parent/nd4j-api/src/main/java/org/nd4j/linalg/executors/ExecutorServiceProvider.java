package org.nd4j.linalg.executors;

import java.util.concurrent.*;

public class ExecutorServiceProvider {

    public static final String EXEC_THREADS = "org.nd4j.parallel.threads";
    public final static String ENABLED = "org.nd4j.parallel.enabled";

    private static final int nThreads;
    private static ExecutorService executorService;
    private static ForkJoinPool forkJoinPool;

    static {
        int defaultThreads = Runtime.getRuntime().availableProcessors();
        boolean enabled = Boolean.parseBoolean(System.getProperty(ENABLED, "true"));
        if (!enabled)
            nThreads = 1;
        else
            nThreads = Integer.parseInt(System.getProperty(EXEC_THREADS, String.valueOf(defaultThreads)));
    }

    public static synchronized ExecutorService getExecutorService() {
        if (executorService != null)
            return executorService;

        executorService = new ThreadPoolExecutor(nThreads, nThreads, 60L, TimeUnit.SECONDS,
                        new LinkedTransferQueue<Runnable>(), new ThreadFactory() {
                            @Override
                            public Thread newThread(Runnable r) {
                                Thread t = Executors.defaultThreadFactory().newThread(r);
                                t.setDaemon(true);
                                return t;
                            }
                        });
        return executorService;
    }

    public static synchronized ForkJoinPool getForkJoinPool() {
        if (forkJoinPool != null)
            return forkJoinPool;
        forkJoinPool = new ForkJoinPool(nThreads, ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
        return forkJoinPool;
    }

}
