package org.deeplearning4j.parallelism;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.deeplearning4j.optimize.solvers.accumulation.FancyBlockingQueue;
import org.junit.Test;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class FancyBlockingQueueTests {

    @Test
    public void testFancyQueue1() throws Exception {
        final FancyBlockingQueue<Integer> queue = new FancyBlockingQueue<>(new LinkedBlockingQueue<Integer>(512), 4);
        long f = 0;
        for(int x = 0; x < 512; x++) {
            queue.add(x);
            f += x;
        }

        assertEquals(512, queue.size());


        final AtomicLong e = new AtomicLong(0);

        Thread[] threads = new Thread[4];
        for (int x = 0; x < 4; x++) {
            final int t = x;
            threads[x] = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (!queue.isEmpty()) {
                        Integer i = queue.poll();
                        //log.info("i: {}", i);
                        e.addAndGet(i);
                    }
                }
            });

            threads[x].start();
        }


        for (int x = 0; x < 4; x++) {
            threads[x].join();
        }

        assertEquals(f * 4, e.get());
    }

    @Test
    public void testFancyQueue2() throws Exception {
        final FancyBlockingQueue<Integer> queue = new FancyBlockingQueue<>(new LinkedBlockingQueue<Integer>(512), 4);
        long f = 0;
        for(int x = 0; x < 512; x++) {
            queue.add(x);
            f += x;
        }

        assertEquals(512, queue.size());


        final AtomicLong e = new AtomicLong(0);

        Thread[] threads = new Thread[4];
        for (int x = 0; x < 4; x++) {
            final int t = x;
            threads[x] = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (!queue.isEmpty()) {
                        Integer i = queue.poll();
                        e.addAndGet(i);

                        try {
                            Thread.sleep(RandomUtils.nextInt(1, 5));
                        } catch (Exception e) {
                            //
                        }
                    }
                }
            });

            threads[x].start();
        }


        for (int x = 0; x < 4; x++) {
            threads[x].join();
        }

        assertEquals(f * 4, e.get());
    }
}
