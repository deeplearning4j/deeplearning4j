package org.nd4j.parameterserver.distributed;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.conf.Configuration;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * This set of tests doesn't has any assertions within.
 * All we care about here - performance and availability
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class VoidParameterServerStressTest {
    private static final int NUM_WORDS = 100000;
    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testPerformanceStandalone() {
        Configuration configuration = Configuration.builder()
                .networkMask("192.168.0.0/16")
                .numberOfShards(1)
                .build();

        configuration.setShardAddresses("192.168.1.36");

        VoidParameterServer.getInstance().init(configuration);
        VoidParameterServer.getInstance().initializeSeqVec(100, NUM_WORDS, 123, 10, true, false);

        final List<Long> times = new CopyOnWriteArrayList<>();

        Thread[] threads = new Thread[8];
        for (int t = 0; t < threads.length; t++) {
            final int e = t;
            threads[t] = new Thread(() -> {
                for (int i = 0; i < 1000; i++) {
                    long time1 = System.nanoTime();
                    INDArray array = VoidParameterServer.getInstance().getVector(RandomUtils.nextInt(0, NUM_WORDS));
                    long time2 = System.nanoTime();

                    times.add(time2 - time1);

                    if (i + 1 % 500 == 0)
                        log.info("Thread {} cnt {}", e, i + 1);
                }
            });
            threads[t].setDaemon(true);
            threads[t].start();
        }


        for (int t = 0; t < threads.length; t++) {
            try {
                threads[t].join();
            } catch (Exception e) { }
        }

        List<Long> newTimes = new ArrayList<>(times);

        Collections.sort(newTimes);

        log.info("p50: {} us", newTimes.get(newTimes.size() / 2) / 1000);
    }
}