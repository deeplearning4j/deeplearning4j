package org.nd4j.parameterserver.distributed;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.*;
import org.nd4j.parameterserver.distributed.messages.aggregations.DotAggregation;
import org.nd4j.parameterserver.distributed.messages.requests.AssignRequestMessage;
import org.nd4j.parameterserver.distributed.messages.requests.VectorRequestMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedAssignMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedDotMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedInitializationMessage;
import org.nd4j.parameterserver.distributed.messages.intercom.DistributedSolidMessage;
import org.nd4j.parameterserver.distributed.transport.MulticastTransport;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class VoidParameterServerTest {
    private static List<String> localIPs;
    private static List<String> badIPs;
    private static final Transport transport = new MulticastTransport();

    @Before
    public void setUp() throws Exception {
        if (localIPs == null) {
            localIPs = new ArrayList<>(VoidParameterServer.getLocalAddresses());

            badIPs = Arrays.asList("127.0.0.1");
        }
    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testNodeRole1() throws Exception {
        final Configuration conf = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(10)
                .multicastNetwork("224.0.1.1")
                .shardAddresses(localIPs)
                .ttl(4)
                .build();

        VoidParameterServer node = new VoidParameterServer();
        node.init(conf, transport);

        assertEquals(NodeRole.SHARD, node.getNodeRole());
        node.shutdown();
    }

    @Test
    public void testNodeRole2() throws Exception {
        final Configuration conf = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(10)
                .shardAddresses(badIPs)
                .backupAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        VoidParameterServer node = new VoidParameterServer();
        node.init(conf, transport);

        assertEquals(NodeRole.BACKUP, node.getNodeRole());
        node.shutdown();
    }

    @Test
    public void testNodeRole3() throws Exception {
        final Configuration conf = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(10)
                .shardAddresses(badIPs)
                .backupAddresses(badIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        VoidParameterServer node = new VoidParameterServer();
        node.init(conf, transport);

        assertEquals(NodeRole.CLIENT, node.getNodeRole());
        node.shutdown();
    }

    @Test
    public void testNodeInitialization1() throws Exception {
        final AtomicInteger failCnt = new AtomicInteger(0);
        final AtomicInteger passCnt = new AtomicInteger(0);

        final Configuration conf = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(10)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        Thread[] threads = new Thread[10];
        for (int t = 0; t < threads.length; t++) {
            threads[t] = new Thread(new Runnable() {
                @Override
                public void run() {
                    VoidParameterServer node = new VoidParameterServer();
                    node.init(conf, transport);

                    if (node.getNodeRole() != NodeRole.SHARD)
                        failCnt.incrementAndGet();

                    passCnt.incrementAndGet();

                    node.shutdown();
                }
            });

            threads[t].start();
        }


        for (int t = 0; t < threads.length; t++) {
            threads[t].join();
        }

        assertEquals(0, failCnt.get());
        assertEquals(threads.length, passCnt.get());
    }

    /**
     * This is very important test, it covers basic messages handling over network.
     * Here we have 1 client, 1 connected Shard + 2 shards available over multicast UDP
     *
     * PLEASE NOTE: This test uses manual stepping through messages
     *
     * @throws Exception
     */
    @Test
    public void testNodeInitialization2() throws Exception {
        final AtomicInteger failCnt = new AtomicInteger(0);
        final AtomicInteger passCnt = new AtomicInteger(0);
        final AtomicInteger startCnt = new AtomicInteger(0);

        INDArray exp = Nd4j.create(new double[]{0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00});


        final Configuration clientConf = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(3)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .streamId(119)
                .forcedRole(NodeRole.CLIENT)
                .ttl(4)
                .build();

        final Configuration shardConf1 = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(3)
                .streamId(119)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        final Configuration shardConf2 = Configuration.builder()
                .unicastPort(34569) // we'll never get anything on this port
                .multicastPort(45678)
                .numberOfShards(3)
                .streamId(119)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        final Configuration shardConf3 = Configuration.builder()
                .unicastPort(34570) // we'll never get anything on this port
                .multicastPort(45678)
                .numberOfShards(3)
                .streamId(119)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();



        VoidParameterServer clientNode = new VoidParameterServer(true);
        clientNode.setShardIndex((short) 0);
        clientNode.init(clientConf);
        clientNode.getTransport().launch(Transport.ThreadingModel.DEDICATED_THREADS);


        assertEquals(NodeRole.CLIENT, clientNode.getNodeRole());


        Thread[] threads = new Thread[3];
        final Configuration[] configurations = new Configuration[]{shardConf1, shardConf2, shardConf3};

        VoidParameterServer[] shards = new VoidParameterServer[threads.length];
        for (int t = 0; t < threads.length; t++) {
            final int x = t;
            threads[t] = new Thread(() -> {


                shards[x] = new VoidParameterServer(true);
                shards[x].setShardIndex((short) x);
                shards[x].init(configurations[x]);

                shards[x].getTransport().launch(Transport.ThreadingModel.DEDICATED_THREADS);

                assertEquals(NodeRole.SHARD, shards[x].getNodeRole());


                startCnt.incrementAndGet();

                passCnt.incrementAndGet();
            });

            threads[t].setDaemon(true);
            threads[t].start();
        }

        // we block until all threads are really started before sending commands
        while (startCnt.get() < threads.length)
            Thread.sleep(500);

        // give additional time to start handlers
        Thread.sleep(1000);

        // now we'll send commands from Client, and we'll check how these messages will be handled
        DistributedInitializationMessage message = DistributedInitializationMessage.builder()
                .numWords(100)
                .columnsPerShard(10)
                .seed(123)
                .useHs(false)
                .useNeg(true)
                .vectorLength(100)
                .build();

        log.info("MessageType: {}", message.getMessageType());

        clientNode.getTransport().sendMessage(message);

        // at this point each and every shard should already have this message

        // now we check message queue within Shards
        for (int t = 0; t < threads.length; t++) {
            VoidMessage incMessage = shards[t].getTransport().takeMessage();
            assertNotEquals("Failed for shard " + t,null, incMessage);
            assertEquals("Failed for shard " + t, message.getMessageType(), incMessage.getMessageType());

            // we should put message back to corresponding
            shards[t].getTransport().putMessage(incMessage);
        }

        /*
            at this moment we're 100% sure that:
                1) Client was able to send message to one of shards
                2) Selected Shard successfully received message from Client
                3) Shard retransmits message to all shards

            Now, we're passing this message to VoidParameterServer manually, and check for execution result
        */

        for (int t = 0; t < threads.length; t++) {
            VoidMessage incMessage = shards[t].getTransport().takeMessage();
            assertNotEquals("Failed for shard " + t,null, incMessage);
            shards[t].handleMessage(message);

            /**
             * Now we're checking how data storage was initialized
             */

            assertEquals(null, shards[t].getNegTable());
            assertEquals(null, shards[t].getSyn1());


            assertNotEquals(null, shards[t].getExpTable());
            assertNotEquals(null, shards[t].getSyn0());
            assertNotEquals(null, shards[t].getSyn1Neg());
        }


        // now we'll check passing for negTable, but please note - we're not sending it right now
        INDArray negTable = Nd4j.create(100000).assign(12.0f);
        DistributedSolidMessage negMessage = new DistributedSolidMessage(WordVectorStorage.NEGATIVE_TABLE, negTable, false);

        for (int t = 0; t < threads.length; t++) {
            shards[t].handleMessage(negMessage);

            assertNotEquals(null, shards[t].getNegTable());
            assertEquals(negTable, shards[t].getNegTable());
        }


        // now we assign each row to something
        for (int t = 0; t < threads.length; t++) {
            shards[t].handleMessage(new DistributedAssignMessage(WordVectorStorage.SYN_0,1, (double) t));

            assertEquals(Nd4j.create(message.getColumnsPerShard()).assign((double) t), shards[t].getSyn0().getRow(1));
        }


        // and now we'll request for aggregated vector for row 1
        clientNode.getVector(1);
        VoidMessage vecm = shards[0].getTransport().takeMessage();

        assertEquals(7, vecm.getMessageType());

        VectorRequestMessage vrm = (VectorRequestMessage) vecm;

        assertEquals(1, vrm.getRowIndex());

        shards[0].handleMessage(vecm);

        Thread.sleep(100);

        // at this moment all 3 shards should already have distributed message
        for (int t = 0; t < threads.length; t++) {
            VoidMessage dm = shards[t].getTransport().takeMessage();

            assertEquals(20, dm.getMessageType());

            shards[t].handleMessage(dm);
        }

        // at this moment we should have messages propagated across all shards
        Thread.sleep(100);

        for (int t = threads.length - 1; t >= 0; t--) {
            VoidMessage msg;
            while ((msg = shards[t].getTransport().takeMessage()) != null) {
                shards[t].handleMessage(msg);
            }
        }

        // and at this moment, Shard_0 should contain aggregated vector for us
        assertEquals(true, shards[0].clipboard.isTracking(1L));
        assertEquals(true, shards[0].clipboard.isReady(1L));

        INDArray jointVector = shards[0].clipboard.nextCandidate().getAccumulatedResult();

        log.info("Joint vector: {}", jointVector);

        assertEquals(exp, jointVector);


        /**
         * now we're going to test real SkipGram round
         */
        // first, we're setting data to something predefined
        for (int t = 0; t < threads.length; t++) {
            shards[t].handleMessage(new DistributedAssignMessage(WordVectorStorage.SYN_0,0, 0.0));
            shards[t].handleMessage(new DistributedAssignMessage(WordVectorStorage.SYN_0,1, 1.0));
            shards[t].handleMessage(new DistributedAssignMessage(WordVectorStorage.SYN_0,2, 2.0));

            shards[t].handleMessage(new DistributedAssignMessage(WordVectorStorage.SYN_1_NEGATIVE,0, 0.0));
            shards[t].handleMessage(new DistributedAssignMessage(WordVectorStorage.SYN_1_NEGATIVE,1, 1.0));
            shards[t].handleMessage(new DistributedAssignMessage(WordVectorStorage.SYN_1_NEGATIVE,2, 2.0));
        }

        DistributedDotMessage ddot = new DistributedDotMessage(2L, WordVectorStorage.SYN_0, WordVectorStorage.SYN_1_NEGATIVE, new int[]{0, 1, 2}, new int[]{0, 1, 2});
        for (int t = 0; t < threads.length; t++) {
            shards[t].handleMessage(ddot);
        }

        Thread.sleep(100);

        for (int t = threads.length - 1; t >= 0; t--) {
            VoidMessage msg;
            while ((msg = shards[t].getTransport().takeMessage()) != null) {
                shards[t].handleMessage(msg);
            }
        }


        // at this moment ot should be caclulated everywhere
        exp = Nd4j.create(new double[]{0.0, 30.0, 120.0});
        for (int t = 0; t< threads.length; t++) {
            assertEquals(true, shards[t].clipboard.isReady(2L));
            DotAggregation dot = (DotAggregation) shards[t].clipboard.unpin(2L);
            INDArray aggregated = dot.getAccumulatedResult();
            assertEquals(exp, aggregated);
        }


        for (int t = 0; t < threads.length; t++) {
            threads[t].join();
        }

        for (int t = 0; t < threads.length; t++) {
            shards[t].shutdown();
        }

        assertEquals(threads.length, passCnt.get());
    }

    /**
     *
     * PLEASE NOTE: This test uses automatic feeding through messages
     *
     * @throws Exception
     */
    @Test
    public void testNodeInitialization3() throws Exception {
        final AtomicInteger failCnt = new AtomicInteger(0);
        final AtomicInteger passCnt = new AtomicInteger(0);
        final AtomicInteger startCnt = new AtomicInteger(0);

        final Configuration clientConf = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(3)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .streamId(119)
                .forcedRole(NodeRole.CLIENT)
                .ttl(4)
                .build();

        final Configuration shardConf1 = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(3)
                .streamId(119)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        final Configuration shardConf2 = Configuration.builder()
                .unicastPort(34569) // we'll never get anything on this port
                .multicastPort(45678)
                .numberOfShards(3)
                .streamId(119)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        final Configuration shardConf3 = Configuration.builder()
                .unicastPort(34570) // we'll never get anything on this port
                .multicastPort(45678)
                .numberOfShards(3)
                .streamId(119)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();



        VoidParameterServer clientNode = new VoidParameterServer();
        clientNode.setShardIndex((short) 0);
        clientNode.init(clientConf);
        clientNode.getTransport().launch(Transport.ThreadingModel.DEDICATED_THREADS);


        assertEquals(NodeRole.CLIENT, clientNode.getNodeRole());


        Thread[] threads = new Thread[3];
        final Configuration[] configurations = new Configuration[]{shardConf1, shardConf2, shardConf3};
        VoidParameterServer[] shards = new VoidParameterServer[threads.length];
        for (int t = 0; t < threads.length; t++) {
            final int x = t;
            threads[t] = new Thread(() -> {
                shards[x] = new VoidParameterServer();
                shards[x].setShardIndex((short) x);
                shards[x].init(configurations[x]);

                shards[x].getTransport().launch(Transport.ThreadingModel.DEDICATED_THREADS);

                assertEquals(NodeRole.SHARD, shards[x].getNodeRole());
                startCnt.incrementAndGet();
            });

            threads[t].setDaemon(true);
            threads[t].start();
        }

        // waiting till all shards are initialized
        while (startCnt.get() < threads.length)
            Thread.sleep(20);


        DistributedInitializationMessage message = DistributedInitializationMessage.builder()
                .numWords(100)
                .columnsPerShard(10)
                .seed(123)
                .useHs(false)
                .useNeg(true)
                .vectorLength(100)
                .build();

        //clientNode.getTransport().sendMessage(message);
        for (int t = 0; t < threads.length; t++) {
            shards[t].handleMessage(message);
        }

        Thread.sleep(200);

        AssignRequestMessage arm = new AssignRequestMessage(WordVectorStorage.SYN_0, 192f,11);
        clientNode.getTransport().sendMessage(arm);

        Thread.sleep(200);

        INDArray vec = clientNode.getVector(11);

        assertEquals(Nd4j.create(30).assign(192f), vec);


        for (int t = 0; t < threads.length; t++) {
            threads[t].join();
        }
    }
}