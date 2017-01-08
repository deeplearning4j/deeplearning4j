package org.nd4j.parameterserver.distributed;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;

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
 * Tests for all environments are paired: one test for blocking messages, other one for non-blocking messages.
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

    /**
     * This test measures performance of blocking messages processing, VectorRequestMessage in this case
     */
    @Test
    public void testPerformanceStandalone1() {
        Configuration configuration = Configuration.builder()
                .networkMask("192.168.0.0/16")
                .numberOfShards(1)
                .build();

        configuration.setShardAddresses("192.168.1.35");

        VoidParameterServer.getInstance().init(configuration);
        VoidParameterServer.getInstance().initializeSeqVec(100, NUM_WORDS, 123, 10, true, false);

        final List<Long> times = new CopyOnWriteArrayList<>();

        Thread[] threads = new Thread[8];
        for (int t = 0; t < threads.length; t++) {
            final int e = t;
            threads[t] = new Thread(() -> {
                List<Long> results = new ArrayList<>();

                int chunk = NUM_WORDS / threads.length;
                int start = e * chunk;
                int end =  (e + 1) * chunk;

                for (int i = 0; i < 1000000; i++) {
                    long time1 = System.nanoTime();
                    INDArray array = VoidParameterServer.getInstance().getVector(RandomUtils.nextInt(start, end));
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if (i + 1 % 500 == 0)
                        log.info("Thread {} cnt {}", e, i + 1);
                }
                times.addAll(results);
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

    /**
     * This test measures performance of non-blocking messages processing, SkipGramRequestMessage in this case
     */
    @Test
    public void testPerformanceStandalone2() {
        Configuration configuration = Configuration.builder()
                .networkMask("192.168.0.0/16")
                .numberOfShards(1)
                .build();

        configuration.setShardAddresses("192.168.1.35");

        VoidParameterServer.getInstance().init(configuration);
        VoidParameterServer.getInstance().initializeSeqVec(100, NUM_WORDS, 123, 10, true, false);

        final List<Long> times = new CopyOnWriteArrayList<>();

        Thread[] threads = new Thread[8];
        for (int t = 0; t < threads.length; t++) {
            final int e = t;
            threads[t] = new Thread(() -> {
                List<Long> results = new ArrayList<>();

                int chunk = NUM_WORDS / threads.length;
                int start = e * chunk;
                int end =  (e + 1) * chunk;

                for (int i = 0; i < 100000; i++) {
                    SkipGramRequestMessage sgrm = getSGRM();
                    long time1 = System.nanoTime();
                    VoidParameterServer.getInstance().execDistributed(sgrm);
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if (i + 1 % 500 == 0)
                        log.info("Thread {} cnt {}", e, i + 1);
                }
                times.addAll(results);
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



    @Test
    public void testPerformanceMulticast1() throws Exception {
        Configuration configuration = Configuration.builder()
                .networkMask("192.168.0.0/16")
                .numberOfShards(1)
                .build();

        configuration.setShardAddresses("192.168.1.35");
        configuration.setForcedRole(NodeRole.CLIENT);

        Configuration[] configurations = new Configuration[10];
        VoidParameterServer[] shards = new VoidParameterServer[10];
        for (int s = 0; s < shards.length; s++) {
            configurations[s] = Configuration.builder()
                    .unicastPort(3789 + s)
                    .networkMask("192.168.0.0/16")
                    .build();

            configurations[s].setShardAddresses("192.168.1.36");

            shards[s] =  new VoidParameterServer(false);
            shards[s].setShardIndex((short) s);
            shards[s].init(configurations[s]);

            assertEquals(NodeRole.SHARD, shards[s].getNodeRole());
        }

        // this is going to be our Client shard
        VoidParameterServer.getInstance().init(configuration);
        assertEquals(NodeRole.CLIENT, VoidParameterServer.getInstance().getNodeRole());

        VoidParameterServer.getInstance().initializeSeqVec(100, NUM_WORDS, 123, 10, true, false);


        final List<Long> times = new CopyOnWriteArrayList<>();

        Thread[] threads = new Thread[8];
        for (int t = 0; t < threads.length; t++) {
            final int e = t;
            threads[t] = new Thread(() -> {
                List<Long> results = new ArrayList<>();

                int chunk = NUM_WORDS / threads.length;
                int start = e * chunk;
                int end =  (e + 1) * chunk;

                for (int i = 0; i < 1000000; i++) {
                    long time1 = System.nanoTime();
                    INDArray array = VoidParameterServer.getInstance().getVector(RandomUtils.nextInt(start, end));
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if (i + 1 % 500 == 0)
                        log.info("Thread {} cnt {}", e, i + 1);
                }
                times.addAll(results);
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

    /**
     * This method just produces random SGRM requests, fot testing purposes.
     * No real sense could be found here.
     *
     * @return
     */
    protected static SkipGramRequestMessage getSGRM() {
        int w1 = RandomUtils.nextInt(0, NUM_WORDS);
        int w2 = RandomUtils.nextInt(0, NUM_WORDS);

        byte[] codes = new byte[RandomUtils.nextInt(0, 50)];
        int[] points = new int[codes.length];
        for (int e = 0; e < codes.length; e++) {
            codes[e] = (byte) (e % 2 == 0 ? 0 : 1);
            points[e] = RandomUtils.nextInt(0, NUM_WORDS);
        }

        return new SkipGramRequestMessage(w1, w2, points, codes, (short) 0, 0.025, 213412L);
    }
}