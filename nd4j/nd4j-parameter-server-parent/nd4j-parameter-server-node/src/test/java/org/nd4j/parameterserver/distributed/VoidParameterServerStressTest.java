/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.parameterserver.distributed;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.sequence.BasicSequenceProvider;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.requests.CbowRequestMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.logic.ClientRouter;
import org.nd4j.parameterserver.distributed.training.impl.CbowTrainer;
import org.nd4j.parameterserver.distributed.training.impl.SkipGramTrainer;
import org.nd4j.parameterserver.distributed.transport.MulticastTransport;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;
import org.nd4j.parameterserver.distributed.transport.Transport;
import org.nd4j.parameterserver.distributed.logic.routing.InterleavedRouter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;

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
    @Ignore
    public void testPerformanceStandalone1() {
        VoidConfiguration voidConfiguration =
                        VoidConfiguration.builder().networkMask("192.168.0.0/16").numberOfShards(1).build();

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
                int end = (e + 1) * chunk;

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
            } catch (Exception e) {
            }
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
    @Ignore
    public void testPerformanceStandalone2() {
        VoidConfiguration voidConfiguration =
                        VoidConfiguration.builder().networkMask("192.168.0.0/16").numberOfShards(1).build();

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
                int end = (e + 1) * chunk;

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
            } catch (Exception e) {
            }
        }

        List<Long> newTimes = new ArrayList<>(times);

        Collections.sort(newTimes);

        log.info("p50: {} us", newTimes.get(newTimes.size() / 2) / 1000);

        parameterServer.shutdown();
    }



    @Test
    @Ignore
    public void testPerformanceMulticast1() throws Exception {
        VoidConfiguration voidConfiguration =
                        VoidConfiguration.builder().networkMask("192.168.0.0/16").numberOfShards(1).build();

        List<String> addresses = new ArrayList<>();
        for (int s = 0; s < 5; s++) {
            addresses.add("192.168.1.35:3789" + s);
        }

        voidConfiguration.setShardAddresses(addresses);
        voidConfiguration.setForcedRole(NodeRole.CLIENT);

        VoidConfiguration[] voidConfigurations = new VoidConfiguration[5];
        VoidParameterServer[] shards = new VoidParameterServer[5];
        for (int s = 0; s < shards.length; s++) {
            voidConfigurations[s] = VoidConfiguration.builder().unicastPort(Integer.valueOf("3789" + s))
                            .networkMask("192.168.0.0/16").build();

            voidConfigurations[s].setShardAddresses(addresses);

            MulticastTransport transport = new MulticastTransport();
            transport.setIpAndPort("192.168.1.35", Integer.valueOf("3789" + s));
            shards[s] = new VoidParameterServer(false);
            shards[s].setShardIndex((short) s);
            shards[s].init(voidConfigurations[s], transport, new SkipGramTrainer());

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
                int end = (e + 1) * chunk;

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
            } catch (Exception e) {
            }
        }

        List<Long> newTimes = new ArrayList<>(times);

        Collections.sort(newTimes);

        log.info("p50: {} us", newTimes.get(newTimes.size() / 2) / 1000);

        parameterServer.shutdown();;

        for (VoidParameterServer server : shards) {
            server.shutdown();
        }
    }

    /**
     * This is one of the MOST IMPORTANT tests
     */
    @Test(timeout = 60000L)
    public void testPerformanceUnicast1() {
        List<String> list = new ArrayList<>();
        for (int t = 0; t < 1; t++) {
            list.add("127.0.0.1:3838" + t);
        }

        VoidConfiguration voidConfiguration = VoidConfiguration.builder().unicastPort(49823).numberOfShards(list.size())
                        .shardAddresses(list).build();

        VoidParameterServer[] shards = new VoidParameterServer[list.size()];
        for (int t = 0; t < shards.length; t++) {
            shards[t] = new VoidParameterServer(NodeRole.SHARD);

            Transport transport = new RoutedTransport();
            transport.setIpAndPort("127.0.0.1", Integer.valueOf("3838" + t));

            shards[t].setShardIndex((short) t);
            shards[t].init(voidConfiguration, transport, new SkipGramTrainer());


            assertEquals(NodeRole.SHARD, shards[t].getNodeRole());
        }

        VoidParameterServer clientNode = new VoidParameterServer(NodeRole.CLIENT);
        RoutedTransport transport = new RoutedTransport();
        ClientRouter router = new InterleavedRouter(0);

        transport.setRouter(router);
        transport.setIpAndPort("127.0.0.1", voidConfiguration.getUnicastPort());

        router.init(voidConfiguration, transport);

        clientNode.init(voidConfiguration, transport, new SkipGramTrainer());
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

                for (int i = 0; i < 200; i++) {
                    long time1 = System.nanoTime();
                    INDArray array = clientNode.getVector(RandomUtils.nextInt(start, end));
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if ((i + 1) % 100 == 0)
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
            } catch (Exception e) {
            }
        }

        List<Long> newTimes = new ArrayList<>(times);

        Collections.sort(newTimes);

        log.info("p50: {} us", newTimes.get(newTimes.size() / 2) / 1000);

        // shutdown everything
        for (VoidParameterServer shard : shards) {
            shard.getTransport().shutdown();
        }

        clientNode.getTransport().shutdown();
    }


    /**
     * This is second super-important test for unicast transport.
     * Here we send non-blocking messages
     */
    @Test
    @Ignore
    public void testPerformanceUnicast2() {
        List<String> list = new ArrayList<>();
        for (int t = 0; t < 5; t++) {
            list.add("127.0.0.1:3838" + t);
        }

        VoidConfiguration voidConfiguration = VoidConfiguration.builder().unicastPort(49823).numberOfShards(list.size())
                        .shardAddresses(list).build();

        VoidParameterServer[] shards = new VoidParameterServer[list.size()];
        for (int t = 0; t < shards.length; t++) {
            shards[t] = new VoidParameterServer(NodeRole.SHARD);

            Transport transport = new RoutedTransport();
            transport.setIpAndPort("127.0.0.1", Integer.valueOf("3838" + t));

            shards[t].setShardIndex((short) t);
            shards[t].init(voidConfiguration, transport, new SkipGramTrainer());


            assertEquals(NodeRole.SHARD, shards[t].getNodeRole());
        }

        VoidParameterServer clientNode = new VoidParameterServer();
        RoutedTransport transport = new RoutedTransport();
        ClientRouter router = new InterleavedRouter(0);

        transport.setRouter(router);
        transport.setIpAndPort("127.0.0.1", voidConfiguration.getUnicastPort());

        router.init(voidConfiguration, transport);

        clientNode.init(voidConfiguration, transport, new SkipGramTrainer());
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

                for (int i = 0; i < 200; i++) {
                    Frame<SkipGramRequestMessage> frame =
                                    new Frame<>(BasicSequenceProvider.getInstance().getNextValue());
                    for (int f = 0; f < 128; f++) {
                        frame.stackMessage(getSGRM());
                    }
                    long time1 = System.nanoTime();
                    clientNode.execDistributed(frame);
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if ((i + 1) % 100 == 0)
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
            } catch (Exception e) {
            }
        }

        List<Long> newTimes = new ArrayList<>(times);

        Collections.sort(newTimes);

        log.info("p50: {} us", newTimes.get(newTimes.size() / 2) / 1000);

        // shutdown everything
        for (VoidParameterServer shard : shards) {
            shard.getTransport().shutdown();
        }

        clientNode.getTransport().shutdown();
    }

    /**
     * This test checks for single Shard scenario, when Shard is also a Client
     *
     * @throws Exception
     */
    @Test(timeout = 60000L)
    public void testPerformanceUnicast3() throws Exception {
        VoidConfiguration voidConfiguration = VoidConfiguration.builder().unicastPort(49823).numberOfShards(1)
                        .shardAddresses(Arrays.asList("127.0.0.1:49823")).build();

        Transport transport = new RoutedTransport();
        transport.setIpAndPort("127.0.0.1", Integer.valueOf("49823"));

        VoidParameterServer parameterServer = new VoidParameterServer(NodeRole.SHARD);
        parameterServer.setShardIndex((short) 0);
        parameterServer.init(voidConfiguration, transport, new CbowTrainer());

        parameterServer.initializeSeqVec(100, NUM_WORDS, 123L, 100, true, false);

        final List<Long> times = new ArrayList<>();

        log.info("Starting loop...");
        for (int i = 0; i < 200; i++) {
            Frame<CbowRequestMessage> frame = new Frame<>(BasicSequenceProvider.getInstance().getNextValue());
            for (int f = 0; f < 128; f++) {
                frame.stackMessage(getCRM());
            }
            long time1 = System.nanoTime();
            parameterServer.execDistributed(frame);
            long time2 = System.nanoTime();

            times.add(time2 - time1);

            if (i % 50 == 0)
                log.info("{} frames passed...", i);
        }


        Collections.sort(times);

        log.info("p50: {} us", times.get(times.size() / 2) / 1000);

        parameterServer.shutdown();
    }

    /**
     * This test checks multiple Clients hammering single Shard
     *
     * @throws Exception
     */
    @Test(timeout = 60000L)
    public void testPerformanceUnicast4() throws Exception {
        VoidConfiguration voidConfiguration = VoidConfiguration.builder().unicastPort(49823).numberOfShards(1)
                        .shardAddresses(Arrays.asList("127.0.0.1:49823")).build();

        Transport transport = new RoutedTransport();
        transport.setIpAndPort("127.0.0.1", Integer.valueOf("49823"));

        VoidParameterServer parameterServer = new VoidParameterServer(NodeRole.SHARD);
        parameterServer.setShardIndex((short) 0);
        parameterServer.init(voidConfiguration, transport, new SkipGramTrainer());

        parameterServer.initializeSeqVec(100, NUM_WORDS, 123L, 100, true, false);


        VoidParameterServer[] clients = new VoidParameterServer[1];
        for (int c = 0; c < clients.length; c++) {
            clients[c] = new VoidParameterServer(NodeRole.CLIENT);

            Transport clientTransport = new RoutedTransport();
            clientTransport.setIpAndPort("127.0.0.1", Integer.valueOf("4872" + c));

            clients[c].init(voidConfiguration, clientTransport, new SkipGramTrainer());

            assertEquals(NodeRole.CLIENT, clients[c].getNodeRole());
        }

        final List<Long> times = new CopyOnWriteArrayList<>();
        log.info("Starting loop...");
        Thread[] threads = new Thread[clients.length];
        for (int t = 0; t < threads.length; t++) {
            final int c = t;
            threads[t] = new Thread(() -> {
                List<Long> results = new ArrayList<>();
                AtomicLong sequence = new AtomicLong(0);
                for (int i = 0; i < 500; i++) {
                    Frame<SkipGramRequestMessage> frame = new Frame<>(sequence.incrementAndGet());
                    for (int f = 0; f < 128; f++) {
                        frame.stackMessage(getSGRM());
                    }
                    long time1 = System.nanoTime();
                    clients[c].execDistributed(frame);
                    long time2 = System.nanoTime();

                    results.add(time2 - time1);

                    if ((i + 1) % 50 == 0)
                        log.info("Thread_{} finished {} frames...", c, i);
                }

                times.addAll(results);
            });

            threads[t].setDaemon(true);
            threads[t].start();
        }



        for (Thread thread : threads)
            thread.join();

        List<Long> newTimes = new ArrayList<>(times);

        Collections.sort(newTimes);

        log.info("p50: {} us", newTimes.get(newTimes.size() / 2) / 1000);

        for (VoidParameterServer client : clients) {
            client.shutdown();
        }

        parameterServer.shutdown();
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

        byte[] codes = new byte[RandomUtils.nextInt(15, 45)];
        int[] points = new int[codes.length];
        for (int e = 0; e < codes.length; e++) {
            codes[e] = (byte) (e % 2 == 0 ? 0 : 1);
            points[e] = RandomUtils.nextInt(0, NUM_WORDS);
        }

        return new SkipGramRequestMessage(w1, w2, points, codes, (short) 0, 0.025, 213412L);
    }


    protected static CbowRequestMessage getCRM() {
        int w1 = RandomUtils.nextInt(0, NUM_WORDS);

        int syn0[] = new int[5];

        for (int e = 0; e < syn0.length; e++) {
            syn0[e] = RandomUtils.nextInt(0, NUM_WORDS);
        }

        byte[] codes = new byte[RandomUtils.nextInt(15, 45)];
        int[] points = new int[codes.length];
        for (int e = 0; e < codes.length; e++) {
            codes[e] = (byte) (e % 2 == 0 ? 0 : 1);
            points[e] = RandomUtils.nextInt(0, NUM_WORDS);
        }

        return new CbowRequestMessage(syn0, points, w1, codes, 0, 0.025, 119);
    }
}
