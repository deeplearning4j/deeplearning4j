package org.deeplearning4j.datasets.parallel;

import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A class for parallelizing loops and providing loop posiiton to current thread.
 */
@Slf4j
public class Parallel {

    public static void For(
            final int loops,
            ExecutorService taskExecutor,
            final Operation operation) {


        final CountDownLatch latch = new CountDownLatch(loops);
        for(int i = 0; i < loops; i++) {
            final int loopNum = i;

            taskExecutor.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        operation.perform(loopNum);
                        latch.countDown();
                    } catch (Exception e) {
                        Logger.getLogger(Parallel.class.getName())
                                .log(Level.SEVERE, "Exception during execution of parallel task", e);
                    }
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException E) {
            log.error("A parallel operation failed.", E);
        }
    }

    public interface Operation {
        void perform(int loopNUm);
    }

}