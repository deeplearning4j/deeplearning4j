package org.nd4j.parameterserver.distributed;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.messages.InitializationMessage;
import org.nd4j.parameterserver.distributed.messages.NegativeBatchMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.transport.LocalTransport;
import org.nd4j.parameterserver.distributed.transport.MulticastTransport;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
     * This is very important test, it covers basic messages handling over network
     *
     * @throws Exception
     */
    @Test
    public void testNodeInitialization2() throws Exception {
        final AtomicInteger failCnt = new AtomicInteger(0);
        final AtomicInteger passCnt = new AtomicInteger(0);
        final AtomicInteger startCnt = new AtomicInteger(0);


        final Configuration clientConf = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(10)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .streamId(119)
                .forcedRole(NodeRole.CLIENT)
                .ttl(4)
                .build();

        final Configuration shardConf1 = Configuration.builder()
                .unicastPort(34567)
                .multicastPort(45678)
                .numberOfShards(10)
                .streamId(119)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        final Configuration shardConf2 = Configuration.builder()
                .unicastPort(34569) // we'll never get anything on this port
                .multicastPort(45678)
                .numberOfShards(10)
                .streamId(119)
                .shardAddresses(localIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();



        VoidParameterServer clientNode = new VoidParameterServer();
        clientNode.init(clientConf);
        clientNode.getTransport().launch(Transport.ThreadingModel.DEDICATED_THREADS);


        assertEquals(NodeRole.CLIENT, clientNode.getNodeRole());


        Thread[] threads = new Thread[2];
        final Configuration[] configurations = new Configuration[]{shardConf1, shardConf2};

        VoidParameterServer[] shards = new VoidParameterServer[threads.length];
        for (int t = 0; t < threads.length; t++) {
            final int x = t;
            threads[t] = new Thread(() -> {


                shards[x] = new VoidParameterServer();
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
            Thread.sleep(100);

        // now we'll send commands from Client, and we'll check how these messages will be handled
        InitializationMessage message = InitializationMessage.builder()
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
        Thread.sleep(100);

        // now we check message queue within Shards
        for (int t = 0; t < threads.length; t++) {
            VoidMessage incMessage = shards[t].getTransport().takeMessage();
            assertNotEquals("Failed for shard " + t,null, incMessage);
            assertEquals("Failed for shard " + t, message.getMessageType(), incMessage.getMessageType());
        }

        /*
            at this moment we're 100% sure that:
                1) Client was able to send message to one of shards
                2) Selected Shard successfully received message from Client
                3) Shard retransmits message to all shards
        */


        for (int t = 0; t < threads.length; t++) {
            threads[t].join();
        }

        assertEquals(threads.length, passCnt.get());
    }
}