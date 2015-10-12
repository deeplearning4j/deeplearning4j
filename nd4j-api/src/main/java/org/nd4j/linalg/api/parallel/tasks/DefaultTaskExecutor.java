package org.nd4j.linalg.api.parallel.tasks;


import java.util.concurrent.*;

/** Default TaskExecutor based on a ThreadPoolExecutor with the
 * number of threads set to the number of processor (cores), as
 * per the Runtime.getRuntime().availableProcessors() method
 */
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
        //Create a fixed thread pool executor, but with daemon threads
        //Use daemon threads so that the thread pool doesn't stop the JVM from shutting down when done
        executorService = Executors.newFixedThreadPool(nThreads, new ThreadFactory() {
            @Override
            public Thread newThread(Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);
                t.setDaemon(true);
                return t;
            }
        });
    }

    @Override
    public <V> Future<V> executeAsync(Task<V> task) {
        return executorService.submit(task);
    }
}
