package org.nd4j.parameterserver.distributed;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.transport.ClientRouter;
import org.nd4j.parameterserver.distributed.transport.MulticastTransport;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;
import org.nd4j.parameterserver.distributed.transport.Transport;
import org.nd4j.parameterserver.distributed.transport.routing.InterleavedRouter;

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
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                .networkMask("192.168.0.0/16")
                .numberOfShards(1)
                .build();

        voidConfiguration.setShardAddresses("192.168.1.35");

        VoidParameterServer parameterServer = new VoidParameterServer();

        parameterServer.init(voidConfiguration);
        parameterServer.initializeSeqVec(100, NUM_WORDS, 123, 10, true, false);

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
                    INDArray array = parameterServer.getVector(RandomUtils.nextInt(start, end));
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if ((i + 1) % 1000 == 0)
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

        parameterServer.shutdown();
    }

    /**
     * This test measures performance of non-blocking messages processing, SkipGramRequestMessage in this case
     */
    @Test
    public void testPerformanceStandalone2() {
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                .networkMask("192.168.0.0/16")
                .numberOfShards(1)
                .build();

        voidConfiguration.setShardAddresses("192.168.1.35");

        VoidParameterServer parameterServer = new VoidParameterServer();

        parameterServer.init(voidConfiguration);
        parameterServer.initializeSeqVec(100, NUM_WORDS, 123, 10, true, false);

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
                    parameterServer.execDistributed(sgrm);
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if ((i + 1) % 1000 == 0)
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

        parameterServer.shutdown();
    }



    @Test
    public void testPerformanceMulticast1() throws Exception {
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                .networkMask("192.168.0.0/16")
                .numberOfShards(1)
                .build();

        List<String> addresses = new ArrayList<>();
        for (int s = 0; s < 5; s++) {
            addresses.add("192.168.1.35:3789" + s);
        }

        voidConfiguration.setShardAddresses(addresses);
        voidConfiguration.setForcedRole(NodeRole.CLIENT);

        VoidConfiguration[] voidConfigurations = new VoidConfiguration[5];
        VoidParameterServer[] shards = new VoidParameterServer[5];
        for (int s = 0; s < shards.length; s++) {
            voidConfigurations[s] = VoidConfiguration.builder()
                    .unicastPort(Integer.valueOf("3789" + s))
                    .networkMask("192.168.0.0/16")
                    .build();

            voidConfigurations[s].setShardAddresses(addresses);

            MulticastTransport transport = new MulticastTransport();
            transport.setIpAndPort("192.168.1.35", Integer.valueOf("3789" + s));
            shards[s] =  new VoidParameterServer(false);
            shards[s].setShardIndex((short) s);
            shards[s].init(voidConfigurations[s], transport);

            assertEquals(NodeRole.SHARD, shards[s].getNodeRole());
        }

        // this is going to be our Client shard
        VoidParameterServer parameterServer = new VoidParameterServer();
        parameterServer.init(voidConfiguration);
        assertEquals(NodeRole.CLIENT, VoidParameterServer.getInstance().getNodeRole());

        log.info("Instantiation finished...");

        parameterServer.initializeSeqVec(100, NUM_WORDS, 123, 20, true, false);


        log.info("Initialization finished...");

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
                    long time1 = System.nanoTime();
                    INDArray array = parameterServer.getVector(RandomUtils.nextInt(start, end));
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if ((i + 1) % 1000 == 0)
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

        parameterServer.shutdown();;

        for (VoidParameterServer server: shards) {
            server.shutdown();
        }
    }

    /**
     * This is one of the MOST IMPORTANT tests
     */
    @Test
    public void testPerformanceUnicast1() {
        List<String> list = new ArrayList<>();
        for (int t = 0; t < 5; t++) {
            list.add("127.0.0.1:3838" + t);
        }

        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                .unicastPort(49823)
                .numberOfShards(list.size())
                .shardAddresses(list)
                .build();

        VoidParameterServer[] shards = new VoidParameterServer[list.size()];
        for (int t = 0; t < shards.length; t++) {
            shards[t] = new VoidParameterServer(NodeRole.SHARD);

            Transport transport = new RoutedTransport();
            transport.setIpAndPort("127.0.0.1",Integer.valueOf("3838" + t));

            shards[t].setShardIndex((short) t);
            shards[t].init(voidConfiguration, transport);


            assertEquals(NodeRole.SHARD, shards[t].getNodeRole());
        }

        VoidParameterServer clientNode = new VoidParameterServer();
        RoutedTransport transport = new RoutedTransport();
        ClientRouter router = new InterleavedRouter(0);

        transport.setRouter(router);
        transport.setIpAndPort("127.0.0.1", voidConfiguration.getUnicastPort());

        router.init(voidConfiguration, transport);

        clientNode.init(voidConfiguration, transport);
        assertEquals(NodeRole.CLIENT, clientNode.getNodeRole());

        final List<Long> times = new CopyOnWriteArrayList<>();

        // at this point, everything should be started, time for tests
        clientNode.initializeSeqVec(100, NUM_WORDS, 123, 25, true, false);

        log.info("Initialization finished, going to tests...");

        Thread[] threads = new Thread[4];
        for (int t = 0; t < threads.length; t++) {
            final int e = t;
            threads[t] = new Thread(() -> {
                List<Long> results = new ArrayList<>();

                int chunk = NUM_WORDS / threads.length;
                int start = e * chunk;
                int end = (e + 1) * chunk;

                for (int i = 0; i < 100000; i++) {
                    long time1 = System.nanoTime();
                    INDArray array = clientNode.getVector(RandomUtils.nextInt(start, end));
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if ((i + 1) % 1000 == 0)
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

        // shutdown everything
        for (VoidParameterServer shard: shards) {
            shard.getTransport().shutdown();
        }

        clientNode.getTransport().shutdown();
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