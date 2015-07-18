package org.nd4j.linalg.api.ops.executioner;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

public class Loop {
    public interface Each {
        void run(int i);
    }

    private static final int CPUs = Runtime.getRuntime().availableProcessors();
    static ExecutorService executor = Executors.newFixedThreadPool(CPUs, new ThreadFactory() {
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }
    });

    public static void withIndex(int start, int stop, final Each body) {
        int chunksize = (stop - start + CPUs - 1) / CPUs;
        int loops = (stop - start + chunksize - 1) / chunksize;
        final CountDownLatch latch = new CountDownLatch(loops);
        for (int i = start; i < stop; ) {
            final int lo = i;
            i += chunksize;
            final int hi = (i < stop) ? i : stop;
            executor.submit(new Runnable() {
                public void run() {
                    for (int i = lo; i < hi; i++)
                        body.run(i);
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
        }
    }
}