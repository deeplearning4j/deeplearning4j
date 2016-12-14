package org.nd4j.parameterserver.distributed;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.transport.LocalTransport;
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
                }
            });

            threads[t].start();
        }


        for (int t = 0; t < threads.length; t++) {
            threads[t].join();
        }

        assertEquals(0, failCnt.get());

    }
}