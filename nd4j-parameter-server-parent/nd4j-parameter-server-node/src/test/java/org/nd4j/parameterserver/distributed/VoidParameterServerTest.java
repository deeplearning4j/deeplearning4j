package org.nd4j.parameterserver.distributed;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.messages.InitializationMessage;
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
                .port(34567)
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
                .port(34567)
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
                .port(34567)
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
                .port(34567)
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

    @Test
    public void testNodeInitialization2() throws Exception {
        final AtomicInteger failCnt = new AtomicInteger(0);
        final AtomicInteger passCnt = new AtomicInteger(0);
        final AtomicInteger startCnt = new AtomicInteger(0);


        final Configuration clientConf = Configuration.builder()
                .port(34567)
                .numberOfShards(10)
                .shardAddresses(badIPs)
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        final Configuration shardConf = Configuration.builder()
                .port(34567)
                .numberOfShards(10)
                .shardAddresses(Collections.singletonList("192.168.1.36"))
                .multicastNetwork("224.0.1.1")
                .ttl(4)
                .build();

        VoidParameterServer clientNode = new VoidParameterServer();
        clientNode.init(clientConf);

        assertEquals(NodeRole.CLIENT, clientNode.getNodeRole());


        Thread[] threads = new Thread[1];
        VoidParameterServer[] shards = new VoidParameterServer[threads.length];
        for (int t = 0; t < threads.length; t++) {
            final int x = t;
            threads[t] = new Thread(() -> {


                shards[x] = new VoidParameterServer();
                shards[x].init(shardConf);

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

        for (int x = 0; x< 100; x++) {
            clientNode.getTransport().sendMessage(message);
            Thread.sleep(500);
        }

        // at this point each and every shard should already have this message
        Thread.sleep(100);

        for (int t = 0; t < threads.length; t++) {
            VoidMessage incMessage = shards[t].getTransport().takeMessage();
            assertNotEquals(null, incMessage);
            assertEquals(message.getMessageType(), incMessage.getMessageType());
        }


        for (int t = 0; t < threads.length; t++) {
            threads[t].join();
        }

        assertEquals(threads.length, passCnt.get());
    }
}