package org.nd4j.linalg.api.parallel.tasks;


import java.util.concurrent.*;

/** Default TaskExecutor based on a
 * (a) ForkJoinPool (for CPU tasks that are ForkJoin RecursiveTask/RecursiveAction, for example)
 * (b) ThreadPoolExecutor (for all other tasks)
 * number of threads set to the number of processor (cores) by default, as per the
 * Runtime.getRuntime().availableProcessors() method
 * Otherwise, set number of threads using "org.nd4j.parallel.cpu.taskexecutorthreads" system property
 */
public class DefaultTaskExecutor implements TaskExecutor {

    public static final String EXEC_THREADS = "org.nd4j.parallel.cpu.taskexecutorthreads";

    private static DefaultTaskExecutor instance;
    private final ExecutorService executorService;
    private final ForkJoinPool forkJoinPool;

    static {
        instance = new DefaultTaskExecutor();
    }

    public static DefaultTaskExecutor getInstance(){
        return instance;
    }

    public DefaultTaskExecutor(){
        int defaultThreads = Runtime.getRuntime().availableProcessors();

        int nThreads = Integer.parseInt(System.getProperty(EXEC_THREADS,String.valueOf(defaultThreads)));

        //Create a fixed thread pool executor, but with daemon threads
        //Use daemon threads so that the thread pool doesn't stop the JVM from shutting down when done
        executorService = new ThreadPoolExecutor(nThreads, nThreads, 60L, TimeUnit.SECONDS,
                new LinkedTransferQueue<Runnable>(),
                new ThreadFactory() {
                    @Override
                    public Thread newThread(Runnable r) {
                        Thread t = Executors.defaultThreadFactory().newThread(r);
                        t.setDaemon(true);
                        return t;
                    }
        });

        forkJoinPool = new ForkJoinPool(nThreads);
    }

    @Override
    public <V> Future<V> executeAsync(Task<V> task) {
        if(task instanceof ForkJoinTask ) {
            forkJoinPool.execute((ForkJoinTask<?>)task);
            return (Future<V>)task;
        } else {
            return executorService.submit(task);
        }
    }
}
